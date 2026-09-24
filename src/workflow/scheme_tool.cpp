#include "json.h"
#include <CommonCrypto/CommonDigest.h>
#include <algorithm>
#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <set>
#include <functional>
#include <tuple>
#include <fcntl.h>
#include <unistd.h>
#include <cerrno>
#include <cstring>
using namespace fgm;
namespace fs=std::filesystem;
struct Resource : std::runtime_error {
    using std::runtime_error::runtime_error;
};
struct Limits {
    uint64_t record=1048576,work=10000000,selection=67108864, scan=268435456;
};
static Limits limits;
static uint64_t recordWork = 0;
static uint64_t scannedBytes = 0;
static uint64_t mul(uint64_t a,uint64_t b) {
    if(b && a>UINT64_MAX/b)throw Resource("checked product overflow");
    return a*b;
}
static uint64_t add(uint64_t a,uint64_t b) {
    if(a>UINT64_MAX-b)throw Resource("checked sum overflow");
    return a+b;
}
static void scan(uint64_t amount) {
    scannedBytes = add(scannedBytes, amount);
    if (scannedBytes > limits.scan) throw Resource("total reads exceed --scan-bytes");
}
static void regularInput(const fs::path &path, bool singleRecord = false) {
    if (!fs::is_regular_file(path)) throw std::runtime_error("input must be a regular file");
    auto bytes = fs::file_size(path);
    if (singleRecord && bytes > limits.record) throw Resource("single-record input exceeds --record-bytes");
    if (bytes > limits.scan - std::min(limits.scan, scannedBytes)) throw Resource("input exceeds remaining --scan-bytes");
}
// One record budget counts allocated reconstruction cells, evaluated circuit
// coefficient terms, tensor summands, and conservative analysis operations.
static void charge(uint64_t amount) {
    recordWork = add(recordWork, amount);
    if (recordWork > limits.work) {
        throw Resource("cumulative record work exceeds --verification-work");
    }
}
static int64_t signedAdd(int64_t a,int64_t b) {
    if((b>0&&a>INT64_MAX-b)||(b<0&&a<INT64_MIN-b))throw Resource("circuit integer overflow");
    return a+b;
}
static uint64_t u64(const std::string&s) {
    if(s.empty()||s.find_first_not_of("0123456789")!=std::string::npos)throw std::runtime_error("expected unsigned decimal");
    uint64_t n=0;
    for(char c:s)n=add(mul(n,10),c-'0');
    return n;
}
static uint32_t positive(const Json &j) {
    auto v=j.num();
    if(v<1||uint64_t(v)>UINT32_MAX)throw std::runtime_error("positive uint32 required");
    return uint32_t(v);
}
static std::string hash(const std::string&s) {
    CC_SHA256_CTX c;
    CC_SHA256_Init(&c);
    size_t offset=0;
    while(offset<s.size()) {
        auto n=std::min<size_t>(s.size()-offset,UINT32_MAX);
        CC_SHA256_Update(&c,s.data()+offset,CC_LONG(n));
        offset+=n;
    }
    unsigned char d[32];
    CC_SHA256_Final(d,&c);
    const char*h="0123456789abcdef";
    std::string out;
    for(auto b:d) {
        out+=h[b>>4];
        out+=h[b&15];
    }
    return out;
}
// Hash exactly the bytes consumed by the parser, without buffering whole files.
class HashBuffer : public std::streambuf {
    std::streambuf *source;
    CC_SHA256_CTX state;
    char pending[4096];
    size_t pendingSize = 0;
    void flush() {
        if (pendingSize) CC_SHA256_Update(&state, pending, CC_LONG(pendingSize));
        pendingSize = 0;
    }
protected:
    int_type underflow() override {
        return source->sgetc();
    }
    int_type uflow() override {
        auto value = source->sbumpc();
        if (!traits_type::eq_int_type(value, traits_type::eof())) {
            scan(1);
            char byte = traits_type::to_char_type(value);
            pending[pendingSize++] = byte;
            if (pendingSize == sizeof(pending)) flush();
        }
        return value;
    }
    std::streamsize xsgetn(char *target, std::streamsize count) override {
        flush();
        auto received = source->sgetn(target, count);
        scan(uint64_t(received));
        std::streamsize offset = 0;
        while (offset < received) {
            auto size = std::min<std::streamsize>(received-offset, UINT32_MAX);
            CC_SHA256_Update(&state, target+offset, CC_LONG(size));
            offset += size;
        }
        return received;
    }
    public:
    explicit HashBuffer(std::streambuf *buffer) : source(buffer) {
        CC_SHA256_Init(&state);
    }
    std::string finish() const {
        auto copy = state;
        if(pendingSize) CC_SHA256_Update(&copy,pending,CC_LONG(pendingSize));
        unsigned char digest[32];
        CC_SHA256_Final(digest, &copy);
        const char *hex = "0123456789abcdef";
        std::string result;
        for (auto byte : digest) {
            result += hex[byte >> 4];
            result += hex[byte & 15];
        }
        return result;
    }
};
struct ParsedHash {
    HashBuffer &buffer;
    std::string *destination;
    ~ParsedHash() {
        if (destination) *destination = buffer.finish();
    }
};
static std::string fileHash(const fs::path&p) {
    regularInput(p);
    std::ifstream f(p,std::ios::binary);
    if(!f)throw std::runtime_error("cannot read source");
    CC_SHA256_CTX c;
    CC_SHA256_Init(&c);
    char buf[65536];
    while(f) {
        f.read(buf,sizeof buf);
        scan(uint64_t(f.gcount()));
        CC_SHA256_Update(&c,buf,CC_LONG(f.gcount()));
    }
    if(!f.eof())throw std::runtime_error("source read failure");
    unsigned char d[32];
    CC_SHA256_Final(d,&c);
    const char*h="0123456789abcdef";
    std::string out;
    for(auto b:d) {
        out+=h[b>>4];
        out+=h[b&15];
    }
    return out;
}
static void le(std::string&s,uint64_t v,unsigned n) {
    for(unsigned i=0; i<n; ++i) {
        s+=char(v&255);
        v>>=8;
    }
}
static void lp(std::string&s,const std::string&v) {
    validUtf8(v);
    if(v.size()>UINT32_MAX)throw Resource("string too long");
    le(s,v.size(),4);
    s+=v;
}
static std::string selectKey(uint64_t seed,const std::string&ns,const std::string&id) {
    std::string s("FGMSELECT\0",10);
    le(s,1,2);
    le(s,seed,8);
    lp(s,"presentation");
    lp(s,ns);
    lp(s,id);
    return hash(s);
}
using Matrix=std::vector<std::vector<int64_t>>;
struct Scheme {
    std::array<uint32_t,3> n;
    uint32_t rank=0;
    bool f2=false;
    std::array<Matrix,3> f;
    bool circuit=false;
    uint64_t operations=0;
    std::string sourceOrientation="cyclic-w";
    Json sourceMetadata = Json::dict();
};
static std::array<uint64_t,3> widths(const Scheme&s) {
    return {
        mul(s.n[0],s.n[1]),mul(s.n[1],s.n[2]),mul(s.n[2],s.n[0])
    };
}
static void shape(const Scheme&s) {
    auto w=widths(s);
    uint64_t total=0;
    for(int p=0; p<3; ++p) {
        if(w[p]>UINT32_MAX)throw Resource("factor width overflow");
        total=add(total,mul(w[p],s.rank));
    }
    if(total>limits.record)throw Resource("coefficient storage exceeds record budget");
    for(int p=0; p<3; ++p) {
        if(s.f[p].size()!=s.rank)throw std::runtime_error("factor rank mismatch");
        for(auto &r:s.f[p]) {
            if(r.size()!=w[p])throw std::runtime_error("factor width mismatch");
            for(auto v:r)if(v>1||v<(s.f2?0:-1))throw std::runtime_error("coefficient outside declared domain");
        }
    }
}
static std::string identity(const Scheme&s,bool canonical) {
    shape(s);
    std::vector<std::vector<int64_t>> terms;
    const int gauges[4][3]= {
        {
            1,1,1
        }, {
            -1,-1,1
        }, {
            -1,1,-1
        }, {
            1,-1,-1
        }
    };
    for(uint32_t r=0; r<s.rank; ++r) {
        std::vector<int64_t> best;
        for(int g=0; g<((canonical&&!s.f2)?4:1); ++g) {
            std::vector<int64_t> t;
            for(int p=0; p<3; ++p)for(auto x:s.f[p][r])t.push_back(x*gauges[g][p]);
            if(g==0||t>best)best=std::move(t);
        }
        terms.push_back(std::move(best));
    }
    if(canonical)std::sort(terms.begin(),terms.end());
    std::string bytes=canonical?std::string("FGMSCHEME\0",10):std::string("FGMFACTORS\0",11);
    le(bytes,1,2);
    le(bytes,s.f2?2:1,1);
    le(bytes,1,1);
    for(auto n:s.n)le(bytes,n,4);
    le(bytes,s.rank,4);
    for(auto &t:terms)for(auto v:t)bytes+=char(uint8_t(v));
    return std::string(canonical?"fgm-scheme-v1:":"fgm-factors-v1:")+hash(bytes);
}
static bool verify(const Scheme&s) {
    shape(s);
    auto w=widths(s);
    uint64_t equations=mul(mul(w[0],w[1]),w[2]);
    charge(mul(equations,s.rank));
    for(uint64_t i=0; i<w[0]; ++i)for(uint64_t j=0; j<w[1]; ++j)for(uint64_t k=0; k<w[2]; ++k) {
        int64_t sum=0;
        for(uint32_t r=0; r<s.rank; ++r)sum+=s.f[0][r][i]*s.f[1][r][j]*s.f[2][r][k];
        if(s.f2)sum&=1;
        int64_t expected=(i%s.n[1]==j/s.n[2] && i/s.n[1]==k%s.n[0] && j%s.n[2]==k/s.n[0]);
        if(sum!=expected)return false;
    }
    return true;
}
static uint64_t naive(const Scheme&s) {
    auto w=widths(s);
    uint64_t cost=0;
    for(int p=0; p<2; ++p)for(auto &r:s.f[p]) {
        uint64_t n=0;
        for(auto v:r)n+=v!=0;
        cost+=n?n-1:0;
    }
    for(uint64_t j=0; j<w[2]; ++j) {
        uint64_t n=0;
        for(auto &r:s.f[2])n+=r[j]!=0;
        cost+=n?n-1:0;
    }
    return cost;
}
static Matrix matrix(const Json&j) {
    if(j.kind!=Json::Array)throw std::runtime_error("factor must be array");
    Matrix m;
    for(auto &r:j.array) {
        if(r.kind!=Json::Array)throw std::runtime_error("factor row must be array");
        std::vector<int64_t> row;
        for(auto &v:r.array)row.push_back(v.num());
        m.push_back(std::move(row));
    }
    return m;
}
static Json matrixJson(const Matrix&m) {
    Json j=Json::list();
    for(auto&r:m) {
        Json row=Json::list();
        for(auto v:r)row.array.emplace_back(v);
        j.array.push_back(std::move(row));
    }
    return j;
}
static Matrix reconstruct(const Json&outputs,const Json&fresh,uint64_t variables,bool f2,uint64_t&ops) {
    if(outputs.kind!=Json::Array||fresh.kind!=Json::Array)throw std::runtime_error("circuit arrays required");
    uint64_t cells=mul(variables,add(add(variables,fresh.array.size()),outputs.array.size()));
    charge(cells);
    Matrix forms;
    for(uint64_t i=0; i<variables; ++i) {
        std::vector<int64_t>r(variables);
        r[i]=1;
        forms.push_back(std::move(r));
    }
    auto linear=[&](const Json&expr) {
        if(expr.kind!=Json::Array)throw std::runtime_error("circuit expression array required");
        std::vector<int64_t>r(variables);
        for(auto&t:expr.array) {
            if(t.kind!=Json::Object||t.object.size()!=2)throw std::runtime_error("circuit term fields must be index,value");
            charge(variables);
            auto index=t.at("index").num(),v=t.at("value").num();
            if(index<0||uint64_t(index)>=forms.size()||(v!=1&&v!=-1))throw std::runtime_error("invalid circuit reference/sign");
            for(uint64_t i=0; i<variables; ++i) {
                int64_t x=forms[index][i];
                if(v==-1) {
                    if(x==INT64_MIN)throw Resource("circuit negation overflow");
                    x=-x;
                }
                r[i]=signedAdd(r[i],x);
                if(f2)r[i]&=1;
            }
        }
        return r;
    };
    for(auto&e:fresh.array) {
        if(e.kind!=Json::Array||e.array.size()!=2)throw std::runtime_error("fresh expression must be binary");
        forms.push_back(linear(e));
        ops=add(ops,1);
    }
    Matrix result;
    for(auto&e:outputs.array) {
        result.push_back(linear(e));
        ops=add(ops,e.array.empty()?0:e.array.size()-1);
    }
    return result;
}
static Scheme fromJson(const Json&j,const std::string&explicitDomain) {
    recordWork = 0;
    Scheme s;
    bool modern=j.has("dimensions");
    if (modern && j.has("n") && dump(j.at("dimensions")) != dump(j.at("n"))) {
        throw std::runtime_error("contradictory dimensions and n");
    }
    if (j.has("rank") && j.has("m") && j.at("rank").num() != j.at("m").num()) {
        throw std::runtime_error("contradictory rank and m");
    }
    for (auto key : {"metadata", "provenance", "evidence", "source_binding", "source_bindings", "artifact_bindings"}) {
        if (j.has(key)) s.sourceMetadata.object[key] = j.at(key);
    }
    if (j.has("artifact_bindings") && j.at("artifact_bindings").kind != Json::Array) {
        throw std::runtime_error("artifact_bindings must be an array");
    }
    auto &dims=j.at(modern?"dimensions":"n");
    if(dims.kind!=Json::Array||dims.array.size()!=3)throw std::runtime_error("three dimensions required");
    for(int i=0; i<3; ++i)s.n[i]=positive(dims.array[i]);
    s.rank=positive(j.at(modern?"rank":"m"));
    std::string domain;
    if(j.has("domain"))domain=j.at("domain").str();
    else {
        auto&z=j.at("z2");
        if(z.kind!=Json::Boolean)throw std::runtime_error("z2 must be Boolean");
        domain=z.boolean?"F2":"ZT";
    }
    if(domain!="ZT"&&domain!="F2")throw std::runtime_error("unsupported coefficient domain");
    if(!explicitDomain.empty()&&explicitDomain!=domain)throw std::runtime_error("explicit domain conflicts with input");
    s.f2=domain=="F2";
    if(j.has("z2")&&(j.at("z2").kind!=Json::Boolean||j.at("z2").boolean!=s.f2))throw std::runtime_error("z2 conflicts with coefficient domain");
    if(j.has("ring")&&j.at("ring").str()!=domain)throw std::runtime_error("ring conflicts with coefficient domain");
    if(j.has("orientation"))s.sourceOrientation=j.at("orientation").str();
    if(s.sourceOrientation!="cyclic-w"&&s.sourceOrientation!="row-major-w")throw std::runtime_error("unsupported orientation; explicit adapter required");
    if(j.has("schema")&&j.at("schema").str()!="fgm-scheme-v1")throw std::runtime_error("unknown scheme schema");
    auto w=widths(s);
    for(int p=0; p<3; ++p)if(w[p]>limits.record)throw Resource("factor width exceeds record budget");
    s.circuit=j.has("u_fresh");
    for(int p=0; p<3; ++p) {
        std::string key(1,"uvw"[p]);
        if(s.circuit) {
            Matrix m=reconstruct(j.at(key),j.at(key+"_fresh"),p==2?s.rank:w[p],s.f2,s.operations);
            if(m.size()!=(p==2?w[p]:s.rank))throw std::runtime_error("incorrect circuit outputs");
            if(p==2) {
                s.f[p].assign(s.rank,std::vector<int64_t>(w[p]));
                for(uint64_t i=0; i<w[p]; ++i)for(uint32_t r=0; r<s.rank; ++r)s.f[p][r][i]=m[i][r];
            }
            else s.f[p]=std::move(m);
        }
        else s.f[p]=matrix(j.at(key));
    }
    shape(s);
    if(s.sourceOrientation=="row-major-w") {
        for(auto &row:s.f[2]) {
            auto before=row;
            for(uint32_t i=0; i<s.n[0]; ++i)for(uint32_t j=0; j<s.n[2]; ++j)row[j*s.n[0]+i]=before[i*s.n[2]+j];
        }
    }
    if (j.has("scheme_id") && j.at("scheme_id").str() != identity(s, true)) {
        throw std::runtime_error("unknown version or mismatched scheme_id");
    }
    if (j.has("factors_id") && j.at("factors_id").str() != identity(s, false)) {
        throw std::runtime_error("unknown version or mismatched factors_id");
    }
    if(s.circuit) {
        auto&cost=j.at("complexity");
        if(cost.at("reduced").num()<0||uint64_t(cost.at("reduced").num())!=s.operations||cost.at("naive").num()<0||uint64_t(cost.at("naive").num())!=naive(s))throw std::runtime_error("circuit operation count mismatch");
    }
    return s;
}
static Json schemeJson(const Scheme&s) {
    Json j=Json::dict();
    j.object["schema"]=Json("fgm-scheme-v1");
    j.object["domain"]=Json(s.f2?"F2":"ZT");
    j.object["orientation"]=Json("cyclic-w");
    Json dims=Json::list();
    for(auto n:s.n)dims.array.emplace_back(int64_t(n));
    j.object["dimensions"]=dims;
    j.object["rank"]=Json(int64_t(s.rank));
    for(int p=0; p<3; ++p)j.object[std::string(1,"uvw"[p])]=matrixJson(s.f[p]);
    return j;
}
static uint64_t factorRank(Matrix m,bool f2) {
    if(m.empty())return 0;
    uint64_t rows=m.size(),cols=m[0].size(),rank=0;
    charge(mul(mul(rows,cols),std::min(rows,cols)));
    int64_t denominator=1;
    for(uint64_t column=0; column<cols&&rank<rows; ++column) {
        uint64_t pivot=rank;
        while(pivot<rows&&!m[pivot][column])++pivot;
        if(pivot==rows)continue;
        std::swap(m[pivot],m[rank]);
        int64_t value=m[rank][column];
        for(uint64_t row=rank+1; row<rows; ++row) {
            for(uint64_t j=column+1; j<cols; ++j) {
                if(f2)m[row][j]^=m[row][column]*m[rank][j];
                else {
                    __int128 first=__int128(value)*m[row][j],second=__int128(m[row][column])*m[rank][j],difference;
                    if(__builtin_sub_overflow(first,second,&difference))throw Resource("factor rank integer overflow");
                    if(difference%denominator)throw std::runtime_error("nonintegral fraction-free elimination");
                    difference/=denominator;
                    if(difference<INT64_MIN||difference>INT64_MAX)throw Resource("factor rank integer overflow");
                    m[row][j]=int64_t(difference);
                }
            }
            m[row][column]=0;
        }
        denominator=value;
        ++rank;
    }
    return rank;
}
static Scheme executionNormalized(const Scheme &source) {
    Scheme result = source;
    if (!result.f2) {
        for (uint32_t term = 0; term < result.rank; ++term) {
            for (int factor = 0; factor < 2; ++factor) {
                auto &row = result.f[factor][term];
                auto first = std::find_if(row.begin(), row.end(), [](auto value) { return value != 0; });
                if (first != row.end() && *first < 0) {
                    for (auto &value : row) value = -value;
                    for (auto &value : result.f[2][term]) value = -value;
                }
            }
        }
    }
    return result;
}
static uint64_t equalPairs(const Matrix &matrix) {
    std::map<std::vector<int64_t>,uint64_t> counts;
    uint64_t pairs = 0;
    for (auto &row : matrix) pairs = add(pairs, counts[row]++);
    return pairs;
}
static Json assess(const Scheme&s) {
    Json j=Json::dict();
    Json coefficientCounts = Json::dict();
    for (int factor = 0; factor < 3; ++factor) {
        std::array<uint64_t,3> counts{};
        for (auto &row : s.f[factor]) for (auto value : row) ++counts[size_t(value+1)];
        Json values = Json::dict();
        values.object["negative"] = Json(int64_t(counts[0]));
        values.object["zero"] = Json(int64_t(counts[1]));
        values.object["positive"] = Json(int64_t(counts[2]));
        coefficientCounts.object[std::string(1,"uvw"[factor])] = values;
    }
    uint64_t zeroTerms = 0;
    for (uint32_t term = 0; term < s.rank; ++term) {
        bool zero = false;
        for (int factor = 0; factor < 3; ++factor) {
            auto &row = s.f[factor][term];
            zero |= std::all_of(row.begin(), row.end(), [](auto value) { return value == 0; });
        }
        zeroTerms += zero;
    }
    j.object["coefficient_counts"] = coefficientCounts;
    j.object["zero_factor_terms"] = Json(int64_t(zeroTerms));
    auto w=widths(s);
    Json ranks=Json::list();
    for(auto &m:s.f)ranks.array.emplace_back(int64_t(factorRank(m,s.f2)));
    j.object["factor_ranks"]=ranks;
    j.object["factor_rank_domain"]=Json(s.f2?"F2":"Q");
    bool normalized=true;
    for(int p=0; p<2; ++p)for(auto &row:s.f[p]) {
        for(auto v:row)if(v) {
            normalized&=v>0;
            break;
        }
    }
    j.object["sign_normalization_status"]=Json(s.f2?"not applicable":normalized?"first nonzero U/V coefficient positive":"not normalized; unchanged");
    Json pairs=Json::list();
    Json sourcePairs=Json::list();
    auto effective = executionNormalized(s);
    bool capacity=s.rank<=350;
    for(auto n:s.n)capacity&=n<=16;
    for(auto width:w)capacity&=width<=64;
    bool candidates=true;
    uint64_t total=0;
    for(int p=0; p<3; ++p) {
        uint64_t pairsHere = equalPairs(effective.f[p]);
        sourcePairs.array.emplace_back(int64_t(equalPairs(s.f[p])));
        pairs.array.emplace_back(int64_t(pairsHere));
        total=add(total,pairsHere);
        candidates&=pairsHere<=500;
    }
    j.object["representation_eligible"]=Json(capacity);
    j.object["candidate_pairs"]=pairs;
    j.object["source_candidate_pairs"]=sourcePairs;
    j.object["candidate_pair_basis"]=Json("positive-first U/V execution normalization");
    j.object["search_eligible"]=Json(capacity&&candidates);
    j.object["potential_pairs"]=Json(int64_t(total));
    bool reducer=!s.f2&&capacity&&candidates;
    if(!capacity) {
        j.object["signed_reducer_eligible"]=Json(false);
        j.object["naive_additions"]=Json(int64_t(naive(s)));
        return j;
    }
    const uint64_t maxReal[3]= {
        32,32,175
    },maxPairs[3]= {
        2016,2016,61075
    };
    Json reals=Json::list(),subs=Json::list();
    for(int p=0; p<3; ++p) {
        Matrix expressions=s.f[p];
        if(p==2) {
            expressions.assign(w[p],std::vector<int64_t>(s.rank));
            for(uint64_t i=0; i<w[p]; ++i)for(uint32_t r=0; r<s.rank; ++r)expressions[i][r]=s.f[p][r][i];
        }
        uint64_t maximum=0;
        std::set<std::tuple<size_t,size_t,int>> signs;
        for(auto&row:expressions) {
            charge(mul(row.size(),row.size()));
            uint64_t nz=0;
            for(auto x:row)nz+=x!=0;
            maximum=std::max(maximum,nz);
            for(size_t a=0; a<row.size(); ++a)if(row[a])for(size_t b=a+1; b<row.size(); ++b)if(row[b])signs.emplace(a,b,row[a]==row[b]?1:-1);
        }
        reals.array.emplace_back(int64_t(maximum));
        subs.array.emplace_back(int64_t(signs.size()));
        reducer&=maximum<=maxReal[p]&&signs.size()<=maxPairs[p];
    }
    j.object["signed_reducer_eligible"]=Json(reducer);
    j.object["reducer_max_real_variables"]=reals;
    j.object["reducer_signed_subexpressions"]=subs;
    j.object["naive_additions"]=Json(int64_t(naive(s)));
    return j;
}
static Json report(const Scheme&s,const std::string&artifact,const Json&binding=Json()) {
    if(!verify(s))throw std::runtime_error("invalid tensor in declared domain");
    Json j=schemeJson(s);
    for (auto &field : s.sourceMetadata.object) j.object[field.first] = field.second;
    Json bindings = j.has("artifact_bindings") ? j.at("artifact_bindings") : Json::list();
    Json current = Json::dict();
    current.object["sha256"] = Json(artifact);
    current.object["role"] = Json("input artifact; earlier bindings are source supplied");
    bindings.array.push_back(current);
    j.object["artifact_bindings"] = bindings;
    j.object["scheme_id"]=Json(identity(s,true));
    j.object["factors_id"]=Json(identity(s,false));
    j.object["source_sha256"]=Json(artifact);
    j.object["normalization"]=Json(s.sourceOrientation=="cyclic-w"?"none; ordered factors preserved":"row-major-w to cyclic-w; term order and signs preserved");
    j.object["source_orientation"]=Json(s.sourceOrientation);
    j.object["submitted_factors_id"]=j.object["factors_id"];
    auto effective = executionNormalized(s);
    j.object["effective_factors_id"]=Json(identity(effective, false));
    Json effectiveFactors = Json::dict();
    for (int p = 0; p < 3; ++p) effectiveFactors.object[std::string(1,"uvw"[p])] = matrixJson(effective.f[p]);
    j.object["effective_factors"]=effectiveFactors;
    j.object["admission_normalization"]=Json("positive-first U/V; source factors preserved; no terms removed");
    j.object["verification"]=Json(s.f2?"exact-F2":"exact-Z");
    j.object["eligibility"]=assess(s);
    j.object["verification_work_used"]=Json(int64_t(recordWork));
    j.object["verification_work_model"]=Json("cumulative reconstruction cells, circuit coefficient terms, tensor summands and conservative analysis operations");
    if(s.circuit)j.object["verified_circuit_additions"]=Json(int64_t(s.operations));
    if (binding.kind != Json::Null) {
        Json origins = j.has("source_bindings") ? j.at("source_bindings") : Json::list();
        if (origins.kind != Json::Array) throw std::runtime_error("source_bindings must be array");
        if (j.has("source_binding")) origins.array.push_back(j.at("source_binding"));
        origins.array.push_back(binding);
        j.object["source_bindings"] = origins;
        j.object["source_binding"] = binding;
    }
    return j;
}
static std::string boundedRead(std::istream&in) {
    std::string s;
    char c;
    while(in.get(c)) {
        if(s.size()>=limits.record)throw Resource("record exceeds --record-bytes");
        s+=c;
    }
    if(!in.eof())throw std::runtime_error("input read failure");
    return s;
}
static std::string jsonFrame(std::istream&in) {
    std::string s;
    int depth=0;
    bool quoted=false,escape=false;
    char c;
    if(!in.get(c))throw std::runtime_error("missing JSON record");
    if(c!='{')throw std::runtime_error("JSON record must be object");
    s+=c;
    depth=1;
    while(depth) {
        if(!in.get(c))throw std::runtime_error("truncated JSON record");
        if(s.size()>=limits.record)throw Resource("record exceeds --record-bytes");
        s+=c;
        if(quoted) {
            if(escape)escape=false;
            else if(c=='\\')escape=true;
            else if(c=='"')quoted=false;
        }
        else if(c=='"')quoted=true;
        else if(c=='{'||c=='[')++depth;
        else if(c=='}'||c==']')--depth;
        if(depth>64)throw Resource("JSON framing depth exceeds 64");
    }
    return s;
}
static void records(const fs::path&path,const std::string&format,const std::string&domain,const std::function<void(const Scheme&,uint64_t)>&emit,uint64_t wanted=UINT64_MAX,std::string *parsedHash=nullptr) {
    regularInput(path, format!="jsonl" && format!="json-array");
    std::ifstream file(path,std::ios::binary);
    if(!file)throw std::runtime_error("cannot open input");
    HashBuffer buffer(file.rdbuf());
    std::istream in(&buffer);
    ParsedHash parsed {
        buffer,parsedHash
    };
    if(!in)throw std::runtime_error("cannot open input");
    uint64_t index=0;
    if(format=="json"||format=="circuit-json") {
        auto j=Parser(boundedRead(in)).parse();
        if(format=="circuit-json"&&!j.has("u_fresh"))throw std::runtime_error("circuit-json requires circuit fields");
        if(wanted==UINT64_MAX||wanted==index)emit(fromJson(j,domain),index);
        return;
    }
    if(format=="jsonl") {
        std::string line;
        char c;
        while(in.get(c)) {
            if(c=='\n') {
                if(line.empty())throw std::runtime_error("empty JSONL record");
                auto j=Parser(line).parse();
                if(wanted==UINT64_MAX||wanted==index)emit(fromJson(j,domain),index);
                ++index;
                line.clear();
            }
            else {
                if(line.size()>=limits.record)throw Resource("JSONL record exceeds limit");
                line+=c;
            }
        }
        if(!line.empty()) {
            auto j=Parser(line).parse();
            if(wanted==UINT64_MAX||wanted==index)emit(fromJson(j,domain),index);
            ++index;
        }
        if(!index)throw std::runtime_error("empty JSONL input");
        return;
    }
    if(format=="json-array") {
        in>>std::ws;
        if(in.get()!='[')throw std::runtime_error("expected JSON array");
        in>>std::ws;
        if(in.peek()==']') {
            in.get();
        }
        else for(; ; ) {
            auto j=Parser(jsonFrame(in)).parse();
            if(wanted==UINT64_MAX||wanted==index)emit(fromJson(j,domain),index);
            ++index;
            in>>std::ws;
            int sep=in.get();
            if(sep==']')break;
            if(sep!=',')throw std::runtime_error("invalid array separator");
            in>>std::ws;
        }
        in>>std::ws;
        if(in.peek()!=EOF)throw std::runtime_error("trailing array content");
        if(!index)throw std::runtime_error("empty scheme array");
        return;
    }
    if(format!="cpu-text"&&format!="metal-search-text"&&format!="metal-minimizer-text")throw std::runtime_error("unsupported input format");
    if(domain!="ZT"&&domain!="F2")throw std::runtime_error("text input requires --domain ZT or F2");
    std::istringstream tokens(boundedRead(in));
    auto token=[&]() {
        std::string t;
        if(!(tokens>>t))throw std::runtime_error("missing text coefficient/header");
        return Parser(t).parse().num();
    };
    auto pos=[&]() {
        auto n=token();
        if(n<1||uint64_t(n)>UINT32_MAX)throw std::runtime_error("invalid text header");
        return uint32_t(n);
    };
    uint32_t count=1;
    if(format=="metal-search-text")count=pos();
    Scheme header;
    if(format=="metal-minimizer-text") {
        for(auto&n:header.n)n=pos();
        header.rank=pos();
        count=pos();
    }
    for(uint32_t i=0; i<count; ++i) {
        recordWork = 0;
        Scheme s;
        s.f2=domain=="F2";
        if(format=="metal-minimizer-text") {
            s.n=header.n;
            s.rank=header.rank;
        }
        else {
            for(auto&n:s.n)n=pos();
            s.rank=pos();
        }
        auto w=widths(s);
        uint64_t cells=mul(s.rank,add(add(w[0],w[1]),w[2]));
        if(cells>limits.record)throw Resource("text coefficient storage exceeds record limit");
        for(int p=0; p<3; ++p) {
            s.f[p].assign(s.rank,std::vector<int64_t>(w[p]));
            for(auto&r:s.f[p])for(auto&v:r)v=token();
        }
        if(wanted==UINT64_MAX||wanted==index) {
            shape(s);
            emit(s,index);
        }
        ++index;
    }
    std::string trailing;
    if(tokens>>trailing)throw std::runtime_error("trailing text tokens");
}
static std::string exportText(const Scheme&s,const std::string&format) {
    std::ostringstream o;
    if(format=="metal-search-text")o<<"1\n";
    o<<s.n[0]<<' '<<s.n[1]<<' '<<s.n[2]<<' '<<s.rank;
    if(format=="metal-minimizer-text")o<<" 1";
    o<<'\n';
    for(auto&m:s.f)for(auto&r:m) {
        for(size_t i=0; i<r.size(); ++i) {
            if(i)o<<' ';
            o<<r[i];
        }
        o<<'\n';
    }
    return o.str();
}
class Output {
    fs::path final,temporary;
    int fd=-1;
    bool installed=false;
    public:
    explicit Output(const fs::path&p):final(p) {
        if(fs::exists(final)||fs::is_symlink(final))throw std::runtime_error("output already exists");
        std::string pattern=(final.parent_path()/("."+final.filename().string()+".XXXXXX")).string();
        std::vector<char>name(pattern.begin(),pattern.end());
        name.push_back(0);
        fd=mkstemp(name.data());
        if(fd<0)throw std::runtime_error("cannot create output temporary");
        temporary=name.data();
    }
    void write(const std::string&s) {
        size_t done=0;
        while(done<s.size()) {
            ssize_t n=::write(fd,s.data()+done,s.size()-done);
            if(n<0&&errno==EINTR)continue;
            if(n<=0)throw std::runtime_error("output write failed");
            done+=size_t(n);
        }
    }
    void commit() {
        if(fsync(fd))throw std::runtime_error("output fsync failed");
        if(::link(temporary.c_str(),final.c_str()))throw std::runtime_error("output publication failed; existing destination preserved");
        installed=true;
    }
    ~Output() {
        if(fd>=0)::close(fd);
        if(!temporary.empty())::unlink(temporary.c_str());
    }
};
static std::string required(const std::map<std::string,std::string>&args,const std::string&key) {
    auto it=args.find(key);
    if(it==args.end())throw std::runtime_error("missing "+key);
    return it->second;
}
static bool within(const fs::path&root,const fs::path&path) {
    auto r=root.begin(),p=path.begin();
    for(; r!=root.end(); ++r,++p)if(p==path.end()||*r!=*p)return false;
    return true;
}
static bool selectedByFilter(const Json &row, const std::map<std::string,std::string> &args) {
    if (args.count("--filter-domain") && row.at("domain").str() != args.at("--filter-domain")) return false;
    if (args.count("--filter-rank")) {
        auto requested = u64(args.at("--filter-rank"));
        if (!row.has("rank") || uint64_t(positive(row.at("rank"))) != requested) return false;
    }
    if (args.count("--filter-dimensions")) {
        auto requested = Parser("["+args.at("--filter-dimensions")+"]").parse();
        if (requested.kind != Json::Array || requested.array.size() != 3) throw std::runtime_error("dimension filter requires three comma-separated positive integers");
        for (auto &value : requested.array) positive(value);
        if (!row.has("dimensions") || dump(row.at("dimensions")) != dump(requested)) return false;
    }
    if (args.count("--filter-group")) {
        auto filter = args.at("--filter-group");
        auto equals = filter.find('=');
        if (equals == std::string::npos || equals == 0 || equals+1 == filter.size() || filter.substr(0,equals).find(':') == std::string::npos) throw std::runtime_error("group filter requires namespace:name=value");
        auto name = filter.substr(0,equals), value = filter.substr(equals+1);
        if (!row.has("metadata") || !row.at("metadata").has("groups") || !row.at("metadata").at("groups").has(name)) return false;
        auto &labels = row.at("metadata").at("groups").at(name);
        if (labels.kind != Json::Array) throw std::runtime_error("group labels must be arrays of strings");
        bool found = false;
        for (auto &label : labels.array) found |= label.str() == value;
        if (!found) return false;
    }
    return true;
}
static void select(const std::map<std::string,std::string>&args,Output&output) {
    fs::path manifest=required(args,"--input");
    regularInput(manifest);
    auto root=fs::canonical(manifest.parent_path().empty()?fs::path("."):manifest.parent_path());
    uint64_t k=u64(required(args,"--count")),seed=args.count("--seed")?u64(args.at("--seed")):0;
    if(k==0)throw std::runtime_error("--count must be positive");
    std::vector<std::string> orderedIds;
    std::map<std::string,size_t> explicitOrder;
    if(args.count("--ids")) {
        if(args.count("--seed"))throw std::runtime_error("--ids conflicts with --seed");
        regularInput(args.at("--ids"), true);
        std::ifstream ids(args.at("--ids"));
        if(!ids)throw std::runtime_error("cannot read IDs");
        auto idBytes=boundedRead(ids);
        scan(idBytes.size());
        auto j=Parser(idBytes).parse();
        if(j.kind!=Json::Array)throw std::runtime_error("--ids requires JSON array");
        for(auto&v:j.array) {
            auto id=v.str();
            if(!explicitOrder.emplace(id,orderedIds.size()).second)throw std::runtime_error("duplicate explicit ID");
            orderedIds.push_back(id);
        }
        if(orderedIds.size()!=k)throw std::runtime_error("--count must match ordered IDs");
    }
    struct Entry {
        std::string key,id;
        Json row;
        size_t order=0;
    };
    auto less=[](const Entry&a,const Entry&b) {
        return a.key!=b.key?a.key<b.key:a.id<b.id;
    };
    std::vector<Entry> selected;
    std::map<std::string,std::string>seen;
    std::string ns;
    uint64_t memory=0;
    std::ifstream manifestFile(manifest);
    if(!manifestFile)throw std::runtime_error("cannot open manifest");
    HashBuffer manifestBuffer(manifestFile.rdbuf());
    std::istream input(&manifestBuffer);
    if(!input)throw std::runtime_error("cannot read collection manifest");
    std::string manifestDigest=fileHash(manifest);
    std::string line;
    uint64_t count=0;
    auto consume=[&]() {
        if(line.empty())throw std::runtime_error("empty collection row");
        Json row=Parser(line).parse();
        if(row.at("schema").str()!="fgm-collection-v1")throw std::runtime_error("unknown collection schema");
        auto rowNs=row.at("namespace").str(),id=row.at("id").str();
        if(rowNs.empty()||id.empty())throw std::runtime_error("empty namespace or source ID");
        if(count==0)ns=rowNs;
        else if(ns!=rowNs)throw std::runtime_error("mixed collection namespaces");
        auto rowDigest=hash(dump(row));
        auto prior=seen.find(id);
        if(prior!=seen.end()) {
            if(prior->second!=rowDigest)throw std::runtime_error("conflicting duplicate collection presentation ID");
            return;
        }
        seen.emplace(id,rowDigest);
        memory=add(memory,id.size()+192);
        if(memory>limits.selection)throw Resource("duplicate registry exceeds --selection-memory");
        ++count;
        if (!selectedByFilter(row,args)) return;
        Entry entry;
        entry.id=id;
        entry.row=row;
        entry.order=explicitOrder.count(id)?explicitOrder[id]:SIZE_MAX;
        entry.key=selectKey(seed,ns,id);
        if(!explicitOrder.empty()&&entry.order==SIZE_MAX)return;
        if(explicitOrder.empty()) {
            selected.push_back(std::move(entry));
            std::push_heap(selected.begin(),selected.end(),less);
            if(selected.size()>k) {
                std::pop_heap(selected.begin(),selected.end(),less);
                selected.pop_back();
            }
        }
        else selected.push_back(std::move(entry));
        uint64_t selectedBytes=0;
        for(auto&e:selected)selectedBytes=add(selectedBytes,dump(e.row).size()+e.id.size()+e.key.size()+128);
        if(add(memory,selectedBytes)>limits.selection)throw Resource("selected records exceed --selection-memory");
    };
    char c;
    while(input.get(c)) {
        if(c=='\n') {
            consume();
            line.clear();
        }
        else {
            if(line.size()>=limits.record)throw Resource("manifest row exceeds record limit");
            line+=c;
        }
    }
    if(!line.empty())consume();
    if(!input.eof())throw std::runtime_error("manifest read failure");
    if(manifestBuffer.finish()!=manifestDigest)throw std::runtime_error("parsed manifest hash mismatch");
    if(selected.size()!=k)throw std::runtime_error("insufficient selected records or missing explicit IDs");
    if(explicitOrder.empty())std::sort(selected.begin(),selected.end(),less);
    else std::sort(selected.begin(),selected.end(),[](auto&a,auto&b) {
        return a.order<b.order;
    });
    for(auto&e:selected) {
        auto &row=e.row;
        fs::path relative=row.at("path").str();
        if(relative.empty()||relative.is_absolute())throw std::runtime_error("source path must be relative");
        for(const auto&p:relative)if(p=="..")throw std::runtime_error("source traversal forbidden");
        auto path=fs::canonical(root/relative);
        if(!within(root,path)||!fs::is_regular_file(path))throw std::runtime_error("source escapes collection root");
        auto sourceFormat=row.at("format").str();
        regularInput(path,sourceFormat!="jsonl"&&sourceFormat!="json-array");
        auto originalHash=fileHash(path);
        if(originalHash!=row.at("sha256").str())throw std::runtime_error("source artifact hash mismatch");
        auto format=row.at("format").str(),domain=row.at("domain").str();
        uint64_t wanted=0;
        if(row.has("locator")) {
            auto&loc=row.at("locator");
            if(loc.kind!=Json::Object||loc.object.size()!=1)throw std::runtime_error("locator requires one index or line");
            if(format=="jsonl") {
                wanted=positive(loc.at("line"))-1;
            }
            else if(format=="json-array"||format=="metal-search-text"||format=="metal-minimizer-text") {
                auto v=loc.at("index").num();
                if(v<0)throw std::runtime_error("negative locator");
                wanted=uint64_t(v);
            }
            else throw std::runtime_error("locator unsupported for single record format");
        }
        else if(format=="jsonl"||format=="json-array"||format=="metal-search-text"||format=="metal-minimizer-text")throw std::runtime_error("multi-record source requires locator");
        bool found=false;
        std::string parsedSourceHash;
        records(path,format,domain,[&](const Scheme&s,uint64_t index) {
            if(index!=wanted)return;
            if(row.has("dimensions")) {
                Json dims=schemeJson(s).at("dimensions");
                if(dump(dims)!=dump(row.at("dimensions")))throw std::runtime_error("manifest dimensions mismatch");
            }
            if(row.has("rank")&&positive(row.at("rank"))!=s.rank)throw std::runtime_error("manifest rank mismatch");
            Json binding=row;
            binding.object["manifest_sha256"]=Json(manifestDigest);
            binding.object["selection_key"]=Json(e.key);
            output.write(dump(report(s,originalHash,binding))+"\n");
            found=true;
        },wanted,&parsedSourceHash);
        if(parsedSourceHash!=originalHash)throw std::runtime_error("parsed source hash mismatch");
        if(!found)throw std::runtime_error("source locator absent");
        if(fileHash(path)!=originalHash)throw std::runtime_error("source changed during selection");
    }
    if(fileHash(manifest)!=manifestDigest)throw std::runtime_error("manifest changed during selection");
}
static void help() {
    std::cout<<"scheme_tool import|export|verify|analyze|select --input FILE --output NEW_FILE\n"
        "  --format cpu-text|metal-search-text|metal-minimizer-text|json|circuit-json|jsonl|json-array\n"
        "  --domain ZT|F2 (required for text; conflicts rejected for JSON)\n"
        "  export: --output-format cpu-text|metal-search-text|metal-minimizer-text|json|legacy-json|jsonl|json-array\n"
        "  select: versioned JSONL manifest; --count K [--seed UINT64 | --ids JSON_FILE]\n"
        "  filters before selection: --filter-domain ZT|F2 --filter-dimensions A,B,C --filter-rank R\n"
        "  --filter-group namespace:name=value matches metadata.groups arrays; absent fields do not match\n"
        "  --record-bytes 1048576 --verification-work 10000000 --selection-memory 67108864 --scan-bytes 268435456\n"
        "Outputs are new files; invalid mathematics exits 1, resource limits exit 2.\n"
        "Verification is exact in the declared domain; no live factor normalization or reduction occurs.\n";
}
int main(int argc,char**argv) {
    try {
        if(argc==2&&std::string(argv[1])=="--help") {
            help();
            return 0;
        }
        if(argc<2)throw std::runtime_error("command required; use --help");
        std::string command=argv[1];
        if(command!="import"&&command!="export"&&command!="verify"&&command!="analyze"&&command!="select")throw std::runtime_error("unknown command");
        std::map<std::string,std::string>args;
        std::set<std::string>allowed= {
            "--input","--output","--format","--domain","--output-format","--record-bytes","--verification-work","--selection-memory","--count","--seed","--ids","--scan-bytes","--filter-domain","--filter-dimensions","--filter-rank","--filter-group"
        };
        for(int i=2; i<argc; i+=2) {
            if(i+1>=argc||!allowed.count(argv[i])||!args.emplace(argv[i],argv[i+1]).second)throw std::runtime_error("unknown, duplicate or missing option");
        }
        if(args.count("--record-bytes"))limits.record=u64(args["--record-bytes"]);
        if(args.count("--verification-work"))limits.work=u64(args["--verification-work"]);
        if(args.count("--selection-memory"))limits.selection=u64(args["--selection-memory"]);
        if(args.count("--scan-bytes"))limits.scan=u64(args["--scan-bytes"]);
        if(!limits.record||!limits.work||!limits.selection||!limits.scan)throw std::runtime_error("resource budgets must be positive");
        if(args.count("--filter-domain")&&args["--filter-domain"]!="ZT"&&args["--filter-domain"]!="F2")throw std::runtime_error("unsupported filter domain");
        if(args.count("--domain")&&args["--domain"]!="ZT"&&args["--domain"]!="F2")throw std::runtime_error("unsupported domain");
        if(command!="select"&&(args.count("--count")||args.count("--seed")||args.count("--ids")||args.count("--filter-domain")||args.count("--filter-dimensions")||args.count("--filter-rank")||args.count("--filter-group")))throw std::runtime_error("selection options require select");
        if(command!="export"&&args.count("--output-format"))throw std::runtime_error("--output-format requires export");
        Output output(required(args,"--output"));
        if(command=="select") {
            if(args.count("--format")||args.count("--domain"))throw std::runtime_error("select uses manifest-declared formats/domains");
            select(args,output);
        }
        else {
            fs::path input=required(args,"--input");
            auto format=required(args,"--format");
            regularInput(input,format!="jsonl"&&format!="json-array");
            auto originalHash=fileHash(input);
            std::string domain=args.count("--domain")?args.at("--domain"):"";
            std::string outformat=command=="export"?required(args,"--output-format"):"jsonl";
            if(outformat!="legacy-json"&&outformat!="jsonl"&&outformat!="json"&&outformat!="json-array"&&outformat!="cpu-text"&&outformat!="metal-search-text"&&outformat!="metal-minimizer-text")throw std::runtime_error("unsupported export format");
            if(outformat=="json-array")output.write("[");
            uint64_t count=0;
            std::string parsedSourceHash;
            records(input,format,domain,[&](const Scheme&s,uint64_t index) {
                Json record=report(s,originalHash);
                record.object["source_record_index"]=Json(int64_t(index));
                if(outformat=="json-array") {
                    if(count)output.write(",");
                    output.write(dump(record));
                }
                else if(outformat=="legacy-json") {
                    if(count)throw std::runtime_error("legacy-json export requires one record");
                    Json legacy=Json::dict();
                    legacy.object["n"]=record.at("dimensions");
                    legacy.object["m"]=record.at("rank");
                    legacy.object["z2"]=Json(s.f2);
                    for(auto key: {
                        "u","v","w"
                    })legacy.object[key]=record.at(key);
                    legacy.object["complexity"]=Json(int64_t(naive(s)));
                    output.write(dump(legacy)+"\n");
                }
                else if(outformat=="jsonl"||outformat=="json") {
                    if(outformat=="json"&&count)throw std::runtime_error("json export requires a single selected record");
                    output.write(dump(record)+"\n");
                }
                else {
                    if(count)throw std::runtime_error("text export requires one record per invocation");
                    output.write(exportText(s,outformat));
                }
                ++count;
            },UINT64_MAX,&parsedSourceHash);
            if(parsedSourceHash!=originalHash)throw std::runtime_error("parsed source hash mismatch");
            if(outformat=="json-array")output.write("]\n");
            if(fileHash(input)!=originalHash)throw std::runtime_error("input changed during processing");
        }
        output.commit();
        return 0;
    }
    catch(const Resource&e) {
        std::cerr<<"resource_limit: "<<e.what()<<'\n';
        return 2;
    }
    catch(const std::exception&e) {
        std::cerr<<"error: "<<e.what()<<'\n';
        return 1;
    }
}
