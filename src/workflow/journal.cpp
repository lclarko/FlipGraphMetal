#include "journal.h"
#include <CommonCrypto/CommonDigest.h>
#include <algorithm>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <set>
#include <array>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>

namespace fgm {
namespace {
namespace fs=std::filesystem;
constexpr size_t headerBytes=168, footerBytes=136, indexRowBytes=65, sortRowBytes=84;
const std::string zeroHash(64,'0'), journalMagic="FGMJNL1\n";
#ifdef FGM_JOURNAL_TESTING
int64_t failWrite=-1,failSync=-1;
size_t capWrite=0;
#endif
[[noreturn]] void ioError(const std::string &what) { throw std::runtime_error(what+": "+std::strerror(errno)); }
uint64_t add(uint64_t a,uint64_t b) { if(b>UINT64_MAX-a)throw Resource("journal size overflow");return a+b; }
uint64_t mul(uint64_t a,uint64_t b) { if(b&&a>UINT64_MAX/b)throw Resource("journal size overflow");return a*b; }
struct Fd {
    int value=-1;
    explicit Fd(int n=-1):value(n){}
    ~Fd(){if(value>=0)::close(value);}
    Fd(const Fd&)=delete;Fd&operator=(const Fd&)=delete;
};
void writeAll(int fd,const std::string &bytes) {
    size_t pos=0;
    while(pos<bytes.size()) {
        size_t amount=bytes.size()-pos;
#ifdef FGM_JOURNAL_TESTING
        if(failWrite==0){errno=EIO;ioError("injected journal write");}
        if(failWrite>0)amount=std::min(amount,size_t(failWrite));
        if(capWrite)amount=std::min(amount,capWrite);
#endif
        auto n=::write(fd,bytes.data()+pos,amount);
        if(n<0&&errno==EINTR)continue;
        if(n<=0)ioError("journal write");
        pos+=size_t(n);
#ifdef FGM_JOURNAL_TESTING
        if(failWrite>0)failWrite-=n;
#endif
    }
}
void syncFile(int fd) {
#ifdef FGM_JOURNAL_TESTING
    if(failSync==0){errno=EIO;ioError("injected journal sync");}
    if(failSync>0)--failSync;
#endif
    if(::fsync(fd))ioError("journal fsync");
#ifdef __APPLE__
    if(::fcntl(fd,F_FULLFSYNC))ioError("journal full sync");
#endif
}
void syncDirectory(const fs::path &path) {
    Fd fd(::open(path.c_str(),O_RDONLY|O_DIRECTORY|O_NOFOLLOW));
    if(fd.value<0||::fsync(fd.value))ioError("journal directory sync");
}
std::string readAt(int fd,uint64_t offset,size_t amount) {
    std::string bytes(amount,'\0');size_t pos=0;
    while(pos<amount) {
        auto n=::pread(fd,bytes.data()+pos,amount-pos,off_t(offset+pos));
        if(n<0&&errno==EINTR)continue;
        if(n<0)ioError("journal read");
        if(!n)throw std::runtime_error("journal changed or truncated during read");
        pos+=size_t(n);
    }
    return bytes;
}
uint64_t fileSize(int fd) {struct stat st{};if(::fstat(fd,&st)||!S_ISREG(st.st_mode)||st.st_size<0)throw std::runtime_error("journal input must be regular");return uint64_t(st.st_size);}
std::string hex(uint64_t n){std::ostringstream s;s<<std::hex<<std::setw(16)<<std::setfill('0')<<n;return s.str();}
uint64_t unhex(const std::string &s){uint64_t n=0;for(char c:s){if(!((c>='0'&&c<='9')||(c>='a'&&c<='f')))throw std::runtime_error("invalid journal hex");n=n*16+(c<='9'?c-'0':c-'a'+10);}return n;}
bool digestText(const std::string &s){return s.size()==64&&s.find_first_not_of("0123456789abcdef")==std::string::npos;}
std::string key(const std::string &id){const std::string prefix="fgm-scheme-v1:";if(id.compare(0,prefix.size(),prefix)||!digestText(id.substr(prefix.size())))throw std::runtime_error("invalid historical scheme identity");return id.substr(prefix.size());}
std::string idOf(const std::string &key){return "fgm-scheme-v1:"+key;}
std::string fileHash(const fs::path &path,const std::function<void(uint64_t)> &account={}) {
    Fd fd(::open(path.c_str(),O_RDONLY|O_NOFOLLOW));if(fd.value<0)ioError("open index hash");
    auto size=fileSize(fd.value);CC_SHA256_CTX state;CC_SHA256_Init(&state);
    for(uint64_t offset=0;offset<size;) {auto amount=size_t(std::min<uint64_t>(65536,size-offset));if(account)account(amount);auto bytes=readAt(fd.value,offset,amount);CC_SHA256_Update(&state,bytes.data(),CC_LONG(bytes.size()));offset+=bytes.size();}
    unsigned char digest[32];CC_SHA256_Final(digest,&state);std::string out;const char *digits="0123456789abcdef";for(auto b:digest){out+=digits[b>>4];out+=digits[b&15];}return out;
}
std::vector<std::string> strings(const Json &j){if(j.kind!=Json::Array)throw std::runtime_error("journal identity list must be array");std::vector<std::string>out;for(auto &v:j.array){key(v.str());out.push_back(v.str());}return out;}
Json list(const std::vector<std::string>&v){Json out=Json::list();for(auto &s:v)out.array.emplace_back(s);return out;}
std::vector<std::pair<std::string,bool>> admissions(const Json &j) {
    auto &a=j.at("admissions");if(a.kind!=Json::Array)throw std::runtime_error("journal admissions must be array");
    std::vector<std::pair<std::string,bool>> result;
    for(auto &v:a.array){auto id=v.at("scheme_id").str();key(id);auto origin=v.at("origin").str();if(origin!="import"&&origin!="discovery")throw std::runtime_error("invalid admission origin");result.emplace_back(id,origin=="discovery");}
    auto sorted=result;std::sort(sorted.begin(),sorted.end());for(size_t i=1;i<sorted.size();++i)if(sorted[i-1].first==sorted[i].first)throw std::runtime_error("duplicate admission within transaction");return result;
}
CommitReceipt receiptOf(const Json &j,uint64_t seq,uint64_t offset,uint64_t bytes,const std::string &hash) {
    auto &m=j.at("_journal");if(m.at("schema").str()!="fgm-journal-accounting-v1")throw std::runtime_error("invalid journal accounting schema");
    CommitReceipt r{seq,offset,bytes,hash,strings(m.at("novel_ids")),strings(m.at("credited_ids"))};
    auto a=admissions(j);std::set<std::string> n(r.novelIds.begin(),r.novelIds.end()),c(r.creditedIds.begin(),r.creditedIds.end());
    if(n.size()!=r.novelIds.size()||c.size()!=r.creditedIds.size())throw std::runtime_error("duplicate journal credit");
    for(auto &v:a){if(c.count(v.first)&&(!n.count(v.first)||!v.second))throw std::runtime_error("invalid journal discovery credit");if(n.count(v.first)&&v.second&&!c.count(v.first))throw std::runtime_error("missing journal discovery credit");n.erase(v.first);c.erase(v.first);}
    if(!n.empty()||!c.empty())throw std::runtime_error("journal accounting references absent admission");return r;
}
}
struct Journal::Impl {
    fs::path dir;JournalLimits limits;Fd lock,file;JournalRecovery current;bool poisoned=false;uint64_t entries=0;std::function<void(uint64_t)> accountRead;
    Impl(const fs::path &path,JournalLimits l,bool create,bool readOnly=false,std::function<void(uint64_t)> account={}):dir(fs::absolute(path)),limits(l),accountRead(std::move(account)) {
        if(!l.storageBytes||!l.maxTransactionBytes||l.maxTransactionBytes>INT64_MAX||l.indexMemoryBytes<1024)throw std::runtime_error("invalid journal limits");
        if(create){if(!fs::create_directory(dir))throw std::runtime_error("journal already exists");syncDirectory(dir.parent_path());}
        if(fs::is_symlink(dir)||!fs::is_directory(dir))throw std::runtime_error("journal directory must be regular");
        if(!readOnly){lock.value=::open((dir/"writer.lock").c_str(),O_RDWR|O_NOFOLLOW|(create?O_CREAT:0),0600);
        if(lock.value<0)ioError("open journal lock");if(::flock(lock.value,LOCK_EX|LOCK_NB))ioError("journal already has writer");}
        file.value=::open((dir/"journal.bin").c_str(),(readOnly?O_RDONLY:O_RDWR)|O_NOFOLLOW|(create?O_CREAT|O_EXCL:0),0600);
        if(file.value<0)ioError("open authoritative journal");
        if(create){writeAll(file.value,journalMagic);syncFile(file.value);syncDirectory(dir);publishHead(0,journalMagic.size(),zeroHash);}
        if(fileSize(file.value)<journalMagic.size()||read(0,journalMagic.size())!=journalMagic)throw std::runtime_error("invalid journal format");
    }
    std::string read(uint64_t offset,size_t amount)const{if(accountRead)accountRead(amount);return readAt(file.value,offset,amount);}
    void healthy()const{if(poisoned)throw std::runtime_error("journal needs close and recovery after failed mutation");}
    uint64_t storage()const{uint64_t total=0;for(auto &p:fs::directory_iterator(dir)){if(p.is_symlink()||!p.is_regular_file())throw std::runtime_error("unexpected nonregular journal entry");total=add(total,p.file_size());}return total;}
    void headroom(uint64_t amount)const{if(add(storage(),amount)>limits.storageBytes)throw Resource("journal storage budget exhausted");}
    void newFile(const fs::path &p,const std::string &bytes){headroom(bytes.size());Fd fd(::open(p.c_str(),O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW,0600));if(fd.value<0)ioError("create journal file");writeAll(fd.value,bytes);syncFile(fd.value);syncDirectory(dir);}
    struct Head {uint64_t sequence,offset;std::string hash;};
    Head readHead() const {
        const auto path=dir/"committed-head.json";
        Fd fd(::open(path.c_str(),O_RDONLY|O_NOFOLLOW));
        if(fd.value<0)throw std::runtime_error("required committed-head watermark is missing or unreadable");
        const auto size=fileSize(fd.value);
        if(size>4096)throw std::runtime_error("invalid committed-head watermark size");
        if(accountRead)accountRead(size);
        auto value=Parser(readAt(fd.value,0,size_t(size))).parse();
        if(value.kind!=Json::Object||value.object.size()!=5||value.at("schema").str()!="fgm-journal-head-v1"||
           value.at("sequence").num()<0||value.at("committed_bytes").num()<int64_t(journalMagic.size()))
            throw std::runtime_error("invalid committed-head watermark");
        const auto checksum=value.at("checksum_sha256").str();value.object.erase("checksum_sha256");
        if(!digestText(checksum)||sha256Bytes(dump(value))!=checksum)
            throw std::runtime_error("committed-head watermark checksum mismatch");
        Head head{uint64_t(value.at("sequence").num()),uint64_t(value.at("committed_bytes").num()),value.at("frame_sha256").str()};
        if(!digestText(head.hash)||(!head.sequence&&(head.offset!=journalMagic.size()||head.hash!=zeroHash))||
           (head.sequence&&head.offset<=journalMagic.size()))throw std::runtime_error("invalid committed-head watermark binding");
        return head;
    }
    void publishHead(uint64_t sequence,uint64_t offset,const std::string &hash) {
        if(sequence>INT64_MAX||offset>INT64_MAX)throw Resource("committed-head integer capacity exceeded");
        Json head=Json::dict();head.object["schema"]=Json("fgm-journal-head-v1");
        head.object["sequence"]=Json(int64_t(sequence));head.object["committed_bytes"]=Json(int64_t(offset));
        head.object["frame_sha256"]=Json(hash);head.object["checksum_sha256"]=Json(sha256Bytes(dump(head)));
        const auto bytes=dump(head)+"\n";headroom(bytes.size());
        auto pattern=(dir/("head-pending-"+hex(sequence)+"-XXXXXX")).string();
        std::vector<char> name(pattern.begin(),pattern.end());name.push_back('\0');
        Fd fd(::mkstemp(name.data()));if(fd.value<0)ioError("create committed-head temporary");
        // Failed writes/syncs leave their exact temporary bytes for inspection.
        // They count toward storage and are never mistaken for acknowledged head.
        writeAll(fd.value,bytes);syncFile(fd.value);
        if(::rename(name.data(),(dir/"committed-head.json").c_str()))ioError("publish committed-head watermark");
        syncDirectory(dir);
    }
    void validateHead() {
        const auto floor=readHead();bool found=!floor.sequence;
        scan([&](const Json&,const CommitReceipt &receipt){
            if(receipt.sequence==floor.sequence) {
                if(receipt.hash!=floor.hash||add(receipt.offset,receipt.bytes)!=floor.offset)
                    throw std::runtime_error("committed-head watermark does not match journal history");
                found=true;
            }
        },false,true);
        if(!found||current.sequence<floor.sequence||current.committedBytes<floor.offset)
            throw std::runtime_error("journal is missing acknowledged committed history");
    }
    void scan(const Visitor &visitor,bool repair,bool allowIncomplete=false) {
        current={0,journalMagic.size(),0,0,zeroHash,{},false};uint64_t size=fileSize(file.value),offset=journalMagic.size();
        for(auto &p:fs::directory_iterator(dir))if(p.path().filename().string().rfind("tail-",0)==0)current.retainedTails.push_back(p.path());
        std::sort(current.retainedTails.begin(),current.retainedTails.end());
        while(offset<size) {
            uint64_t remaining=size-offset,total=0,sequence=0;std::string header;
            if(remaining>=headerBytes){header=read(offset,headerBytes);if(header.substr(0,8)!="FGMTX001"||sha256Bytes(header.substr(0,104))!=header.substr(104,64))throw std::runtime_error("journal header corruption");sequence=unhex(header.substr(8,16));auto count=unhex(header.substr(24,16));if(sequence!=current.sequence+1||header.substr(40,64)!=current.hash)throw std::runtime_error("journal sequence/hash chain corruption");if(count>limits.maxTransactionBytes)throw Resource("journal transaction exceeds byte limit");total=add(add(headerBytes,count),footerBytes);}
            if(remaining<headerBytes||remaining<total){current.incompleteTailBytes=remaining;if(allowIncomplete&&!repair)break;if(!repair)throw std::runtime_error("unexpected incomplete journal tail");headroom(remaining);auto tail=read(offset,size_t(remaining));auto path=dir/("tail-"+hex(offset)+"-"+sha256Bytes(tail)+".bin");if(!fs::exists(path))newFile(path,tail);else if(fileHash(path)!=sha256Bytes(tail))throw std::runtime_error("tail evidence collision");if(::ftruncate(file.value,off_t(offset)))ioError("truncate incomplete journal tail");syncFile(file.value);current.incompleteTailBytes=0;if(std::find(current.retainedTails.begin(),current.retainedTails.end(),path)==current.retainedTails.end())current.retainedTails.push_back(path);std::sort(current.retainedTails.begin(),current.retainedTails.end());break;}
            auto payload=read(offset+headerBytes,size_t(total-headerBytes-footerBytes));auto footer=read(offset+total-footerBytes,footerBytes);auto hash=sha256Bytes(header+sha256Bytes(payload));if(footer!="FGMEND01"+sha256Bytes(payload)+hash)throw std::runtime_error("committed journal checksum corruption");auto transaction=Parser(payload).parse();auto receipt=receiptOf(transaction,sequence,offset,total,hash);current.admissionCount=add(current.admissionCount,admissions(transaction).size());if(visitor)visitor(transaction,receipt);offset+=total;current.sequence=sequence;current.hash=hash;current.committedBytes=offset;
        }
    }
    bool indexValid() {
        try {
            auto metaPath=dir/"index.json",dataPath=dir/"index.ids";
            if(fs::is_symlink(metaPath)||fs::is_symlink(dataPath)||!fs::is_regular_file(metaPath)||!fs::is_regular_file(dataPath)||fs::file_size(metaPath)>4096)return false;
            if(accountRead)accountRead(fs::file_size(metaPath));std::ifstream in(metaPath);std::string bytes((std::istreambuf_iterator<char>(in)),{});auto m=Parser(bytes).parse();if(m.at("schema").str()!="fgm-journal-index-v1"||m.at("sequence").num()<0||uint64_t(m.at("sequence").num())!=current.sequence||m.at("head_sha256").str()!=current.hash||m.at("count").num()<0)return false;entries=uint64_t(m.at("count").num());if(fs::file_size(dataPath)!=mul(entries,indexRowBytes)||m.at("index_sha256").str()!=fileHash(dataPath,accountRead))return false;
            if(accountRead)accountRead(fs::file_size(dataPath));std::ifstream data(dataPath);std::string line,previous;uint64_t count=0;while(std::getline(data,line)){if(!digestText(line)||(!previous.empty()&&line<=previous))return false;previous=line;++count;}return count==entries;
        }catch(const Resource&){throw;}catch(const std::exception&){return false;}
    }
    void publishIndex(const fs::path &temporary,uint64_t count) {
        Json meta=Json::dict();meta.object["schema"]=Json("fgm-journal-index-v1");meta.object["sequence"]=Json(int64_t(current.sequence));meta.object["head_sha256"]=Json(current.hash);meta.object["count"]=Json(int64_t(count));meta.object["index_sha256"]=Json(fileHash(temporary));
        auto tempMeta=dir/"scratch-index.json";if(fs::exists(tempMeta))throw std::runtime_error("unexpected index scratch");newFile(tempMeta,dump(meta)+"\n");fs::rename(temporary,dir/"index.ids");syncDirectory(dir);fs::rename(tempMeta,dir/"index.json");syncDirectory(dir);entries=count;
    }
    void clearScratch(){for(auto &p:fs::directory_iterator(dir)){auto name=p.path().filename().string();if(name.rfind("scratch-",0)==0){if(p.is_symlink()||!p.is_regular_file())throw std::runtime_error("invalid index scratch");fs::remove(p.path());}}syncDirectory(dir);}
    void rebuild() {
        clearScratch();std::array<fs::path,64> levels{};std::vector<std::string> rows;
        const uint64_t perChunk=std::max<uint64_t>(1,limits.indexMemoryBytes/(2*sortRowBytes+sizeof(std::string)+64));uint64_t serial=0,totalRows=0;
        auto chunkPath=[&](){return dir/("scratch-sort-"+hex(serial++));};
        auto merge=[&](const fs::path &left,const fs::path &right){
            auto path=chunkPath();headroom(add(fs::file_size(left),fs::file_size(right)));
            Fd out(::open(path.c_str(),O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW,0600));if(out.value<0)ioError("create index merge");
            std::ifstream a(left),b(right);if(!a||!b)throw std::runtime_error("cannot read index chunks");
            std::string x,y;bool ax=bool(std::getline(a,x)),by=bool(std::getline(b,y));
            while(ax||by){if(!by||(ax&&x<=y)){writeAll(out.value,x+'\n');ax=bool(std::getline(a,x));}else{writeAll(out.value,y+'\n');by=bool(std::getline(b,y));}}
            if(!a.eof()||!b.eof())throw std::runtime_error("index chunk read failed");
            syncFile(out.value);fs::remove(left);fs::remove(right);return path;
        };
        auto flush=[&](){if(rows.empty())return;std::sort(rows.begin(),rows.end());auto path=chunkPath();std::string bytes;for(auto &row:rows)bytes+=row+'\n';newFile(path,bytes);rows.clear();size_t level=0;while(level<levels.size()&&!levels[level].empty()){path=merge(levels[level],path);levels[level++].clear();}if(level==levels.size())throw Resource("index merge level capacity exceeded");levels[level]=path;};
        scan([&](const Json &j,const CommitReceipt&r){std::set<std::string> novel(r.novelIds.begin(),r.novelIds.end()),credited(r.creditedIds.begin(),r.creditedIds.end());for(auto &a:admissions(j)){rows.push_back(key(a.first)+hex(r.sequence)+(novel.count(a.first)?'1':'0')+(credited.count(a.first)?'1':'0')+(a.second?'1':'0'));totalRows=add(totalRows,1);if(rows.size()>=perChunk)flush();}},false);flush();
        fs::path chunk;
        for(auto &level:levels)if(!level.empty())chunk=chunk.empty()?level:merge(chunk,level);
        auto outPath=dir/"scratch-index.ids";headroom(mul(totalRows,indexRowBytes));Fd out(::open(outPath.c_str(),O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW,0600));if(out.value<0)ioError("create historical index");uint64_t count=0;std::string previous,row;
        if(!chunk.empty()){std::ifstream input(chunk);if(!input)throw std::runtime_error("cannot read final index chunk");while(std::getline(input,row)){if(row.size()!=sortRowBytes-1)throw std::runtime_error("invalid index sort row");auto id=row.substr(0,64);bool first=id!=previous;if((row[80]=='1')!=first||((row[81]=='1')!=(first&&row[82]=='1')))throw std::runtime_error("journal historical credit mismatch");if(first){writeAll(out.value,id+'\n');++count;}previous=id;}if(!input.eof())throw std::runtime_error("final index chunk read failed");}
        syncFile(out.value);publishIndex(outPath,count);if(!chunk.empty())fs::remove(chunk);syncDirectory(dir);
    }
    bool contains(const std::string &id)const{auto k=key(id);Fd in(::open((dir/"index.ids").c_str(),O_RDONLY|O_NOFOLLOW));if(in.value<0)ioError("open historical index");uint64_t left=0,right=entries;while(left<right){auto mid=left+(right-left)/2;if(accountRead)accountRead(indexRowBytes);auto row=readAt(in.value,mid*indexRowBytes,indexRowBytes);auto value=row.substr(0,64);if(k==value)return true;if(value<k)left=mid+1;else right=mid;}return false;}
    void mergeIndex(const std::vector<std::string> &newIds){auto ids=newIds;std::sort(ids.begin(),ids.end());auto path=dir/"scratch-index.ids";headroom(mul(add(entries,ids.size()),indexRowBytes)+4096);Fd out(::open(path.c_str(),O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW,0600));if(out.value<0)ioError("create index update");std::ifstream in(dir/"index.ids");if(!in)throw std::runtime_error("cannot read current historical index");std::string line;bool got=bool(std::getline(in,line));size_t next=0;while(got||next<ids.size()){if(next==ids.size()||(got&&line<key(ids[next]))){writeAll(out.value,line+'\n');got=bool(std::getline(in,line));}else{writeAll(out.value,key(ids[next++])+'\n');}}if(!in.eof())throw std::runtime_error("historical index read failed");syncFile(out.value);publishIndex(path,add(entries,ids.size()));}
};
Journal::Journal(const fs::path&p,JournalLimits l,bool create):impl(std::make_unique<Impl>(p,l,create)){recover();}
Journal::~Journal()=default;
JournalRecovery Journal::recover(const Visitor &visitor){impl->healthy();try{impl->validateHead();impl->scan({},true);impl->rebuild();impl->publishHead(impl->current.sequence,impl->current.committedBytes,impl->current.hash);if(visitor)impl->scan(visitor,false);impl->current.derivedIndexCurrent=true;return impl->current;}catch(...){impl->poisoned=true;throw;}}
JournalRecovery Journal::replayReadOnly(const fs::path &path,JournalLimits limits,const Visitor &visitor,const std::function<void(uint64_t)> &accountRead){
    Impl read(path,limits,false,true,accountRead);const auto before=fileSize(read.file.value);
    read.validateHead();const auto expected=read.current;bool indexMatches=read.indexValid();uint64_t historicalCount=0;
    // A derived view never establishes authority. Validate credit through
    // bounded prefix scans when writes/external sorting are forbidden.
    read.scan([&](const Json &j,const CommitReceipt &receipt){
        std::set<std::string> novel(receipt.novelIds.begin(),receipt.novelIds.end());
        for(const auto &candidate:admissions(j)){
            bool seen=false;
            for(uint64_t offset=journalMagic.size();offset<receipt.offset&&!seen;){
                const auto header=read.read(offset,headerBytes);
                const auto count=unhex(header.substr(24,16));
                if(count>limits.maxTransactionBytes)throw Resource("journal transaction exceeds byte limit");
                const auto prior=Parser(read.read(offset+headerBytes,size_t(count))).parse();
                for(const auto &entry:admissions(prior))if(entry.first==candidate.first){seen=true;break;}
                offset=add(offset,add(count,headerBytes+footerBytes));
            }
            if(bool(novel.count(candidate.first))==seen)throw std::runtime_error("journal historical credit mismatch");
            if(!seen){historicalCount=add(historicalCount,1);if(indexMatches&&!read.contains(candidate.first))indexMatches=false;}
        }
    },false,true);
    if(visitor)read.scan(visitor,false,true);
    if(fileSize(read.file.value)!=before||read.current.sequence!=expected.sequence||read.current.hash!=expected.hash)
        throw std::runtime_error("journal changed during read-only replay");
    read.current.derivedIndexCurrent=indexMatches&&historicalCount==read.entries;
    return read.current;
}
JournalRecovery Journal::state()const{impl->healthy();return impl->current;}
bool Journal::contains(const std::string&id)const{impl->healthy();return impl->contains(id);}
void Journal::rebuildIndex(){recover();}
Journal::Reservation Journal::reserve(uint64_t bytes,uint64_t extra){impl->healthy();if(bytes>impl->limits.maxTransactionBytes)throw Resource("journal transaction reservation exceeds limit");auto newRows=bytes/64+1;auto scratch=add(mul(add(impl->entries,newRows),indexRowBytes),mul(add(impl->current.admissionCount,newRows),2*sortRowBytes));auto needed=add(add(mul(add(bytes,headerBytes+footerBytes),2),8192),add(scratch,extra));impl->headroom(needed);Reservation r;r.payloadBytes=bytes;r.storageBytes=needed;r.sequence=impl->current.sequence;r.owner=this;return r;}
CommitReceipt Journal::append(const Json &transaction,const Reservation &reservation){impl->healthy();const auto floor=impl->readHead();if(floor.sequence!=impl->current.sequence||floor.offset!=impl->current.committedBytes||floor.hash!=impl->current.hash)throw std::runtime_error("committed-head watermark changed before append");if(reservation.owner!=this||reservation.sequence!=impl->current.sequence)throw std::runtime_error("invalid or stale journal reservation");if(transaction.kind!=Json::Object||transaction.has("_journal"))throw std::runtime_error("transaction must be object without reserved _journal field");auto admitted=admissions(transaction);CommitReceipt receipt;for(auto &entry:admitted)if(!impl->contains(entry.first)){receipt.novelIds.push_back(entry.first);if(entry.second)receipt.creditedIds.push_back(entry.first);}Json payload=transaction,account=Json::dict();account.object["schema"]=Json("fgm-journal-accounting-v1");account.object["novel_ids"]=list(receipt.novelIds);account.object["credited_ids"]=list(receipt.creditedIds);payload.object["_journal"]=std::move(account);auto bytes=dump(payload);if(bytes.size()>reservation.payloadBytes||bytes.size()>impl->limits.maxTransactionBytes)throw Resource("journal payload exceeds reservation");impl->headroom(reservation.storageBytes);receipt.sequence=impl->current.sequence+1;if(receipt.sequence>INT64_MAX)throw Resource("journal sequence exhausted");receipt.offset=impl->current.committedBytes;auto prefix="FGMTX001"+hex(receipt.sequence)+hex(bytes.size())+impl->current.hash;auto header=prefix+sha256Bytes(prefix);receipt.hash=sha256Bytes(header+sha256Bytes(bytes));auto frame=header+bytes+"FGMEND01"+sha256Bytes(bytes)+receipt.hash;receipt.bytes=frame.size();try{if(::lseek(impl->file.value,0,SEEK_END)!=off_t(receipt.offset))throw std::runtime_error("journal changed before append");writeAll(impl->file.value,frame);syncFile(impl->file.value);impl->current.sequence=receipt.sequence;impl->current.hash=receipt.hash;impl->current.committedBytes=add(receipt.offset,receipt.bytes);impl->current.admissionCount=add(impl->current.admissionCount,admitted.size());impl->mergeIndex(receipt.novelIds);impl->publishHead(receipt.sequence,impl->current.committedBytes,receipt.hash);impl->current.derivedIndexCurrent=true;return receipt;}catch(...){impl->poisoned=true;throw;}}
#ifdef FGM_JOURNAL_TESTING
void Journal::testFaults(int64_t write,int64_t sync,size_t cap){failWrite=write;failSync=sync;capWrite=cap;}
#endif
}
