#pragma once
#include <cstdint>
#include <map>
#include <string>
#include <vector>
#include <stdexcept>
#include <limits>
namespace fgm {
    struct Json {
        enum Kind {
            Null, Boolean, Integer, Decimal, String, Array, Object
        }
        kind=Null;
        bool boolean=false;
        int64_t integer=0;
        std::string string;
        std::vector<Json> array;
        std::map<std::string,Json> object;
        Json()=default;
        explicit Json(bool x):kind(Boolean),boolean(x) {
        }
        explicit Json(int64_t x):kind(Integer),integer(x) {
        }
        explicit Json(std::string x):kind(String),string(std::move(x)) {
        }
        explicit Json(const char *x):Json(std::string(x)) {
        }
        static Json list() {
            Json j;
            j.kind=Array;
            return j;
        }
        static Json dict() {
            Json j;
            j.kind=Object;
            return j;
        }
        const Json &at(const std::string &key) const {
            if(kind!=Object || !object.count(key)) throw std::runtime_error("missing field: "+key);
            return object.at(key);
        }
        bool has(const std::string &key) const {
            return kind==Object && object.count(key);
        }
        std::string str() const {
            if(kind!=String)throw std::runtime_error("expected string");
            return string;
        }
        int64_t num() const {
            if(kind!=Integer)throw std::runtime_error("expected integer, not Boolean or decimal");
            return integer;
        }
    };
    inline void utf8(std::string &s,uint32_t c) {
        if(c<128)s+=char(c);
        else if(c<2048) {
            s+=char(0xc0|(c>>6));
            s+=char(0x80|(c&63));
        }
        else if(c<65536) {
            s+=char(0xe0|(c>>12));
            s+=char(0x80|((c>>6)&63));
            s+=char(0x80|(c&63));
        }
        else {
            s+=char(0xf0|(c>>18));
            s+=char(0x80|((c>>12)&63));
            s+=char(0x80|((c>>6)&63));
            s+=char(0x80|(c&63));
        }
    }
    inline void validUtf8(const std::string &s) {
        for(size_t i=0; i<s.size(); ) {
            unsigned char c=s[i++];
            if(c<128)continue;
            unsigned n=0;
            uint32_t value=0,minimum=0;
            if(c>=0xc2 && c<=0xdf) {
                n=1;
                value=c&31;
                minimum=128;
            }
            else if(c>=0xe0 && c<=0xef) {
                n=2;
                value=c&15;
                minimum=2048;
            }
            else if(c>=0xf0 && c<=0xf4) {
                n=3;
                value=c&7;
                minimum=65536;
            }
            else throw std::runtime_error("invalid UTF-8");
            if(i+n>s.size())throw std::runtime_error("truncated UTF-8");
            while(n--) {
                unsigned char d=s[i++];
                if((d&0xc0)!=0x80)throw std::runtime_error("invalid UTF-8 continuation");
                value=(value<<6)|(d&63);
            }
            if(value<minimum || value>0x10ffff || (value>=0xd800&&value<=0xdfff))throw std::runtime_error("invalid UTF-8 scalar");
        }
    }
    class Parser {
        const std::string &s;
        size_t i=0;
        [[noreturn]] void bad() const {
            throw std::runtime_error("invalid JSON at byte "+std::to_string(i));
        }
        void ws() {
            while(i<s.size() && (s[i]==' '||s[i]=='\t'||s[i]=='\r'||s[i]=='\n'))++i;
        }
        uint32_t hex() {
            uint32_t x=0;
            for(int k=0; k<4; ++k) {
                if(i>=s.size())bad();
                char c=s[i++];
                x*=16;
                if(c>='0'&&c<='9')x+=c-'0';
                else if(c>='a'&&c<='f')x+=c-'a'+10;
                else if(c>='A'&&c<='F')x+=c-'A'+10;
                else bad();
            }
            return x;
        }
        std::string quoted() {
            if(i>=s.size()||s[i++]!='"')bad();
            std::string out;
            while(i<s.size()) {
                unsigned char c=s[i++];
                if(c=='"') {
                    validUtf8(out);
                    return out;
                }
                if(c<32)bad();
                if(c!='\\') {
                    out+=char(c);
                    continue;
                }
                if(i>=s.size())bad();
                char e=s[i++];
                if(e=='"'||e=='\\'||e=='/')out+=e;
                else if(e=='b')out+='\b';
                else if(e=='f')out+='\f';
                else if(e=='n')out+='\n';
                else if(e=='r')out+='\r';
                else if(e=='t')out+='\t';
                else if(e=='u') {
                    uint32_t x=hex();
                    if(x>=0xd800&&x<=0xdbff) {
                        if(i+2>s.size()||s[i++]!='\\'||s[i++]!='u')bad();
                        uint32_t y=hex();
                        if(y<0xdc00||y>0xdfff)bad();
                        x=0x10000+((x-0xd800)<<10)+(y-0xdc00);
                    }
                    else if(x>=0xdc00&&x<=0xdfff)bad();
                    utf8(out,x);
                }
                else bad();
            }
            bad();
        }
        Json value(unsigned depth) {
            if(depth>64)throw std::runtime_error("JSON depth exceeds 64");
            ws();
            if(i>=s.size())bad();
            char c=s[i];
            if(c=='"')return Json(quoted());
            if(c=='{'||c=='[') {
                ++i;
                Json out=c=='{'?Json::dict():Json::list();
                char close=c=='{'?'}':']';
                ws();
                if(i<s.size()&&s[i]==close) {
                    ++i;
                    return out;
                }
                for(; ; ) {
                    ws();
                    if(c=='{') {
                        auto key=quoted();
                        ws();
                        if(i>=s.size()||s[i++]!=':')bad();
                        auto child=value(depth+1);
                        if(!out.object.emplace(key,std::move(child)).second)throw std::runtime_error("duplicate JSON key: "+key);
                    }
                    else out.array.push_back(value(depth+1));
                    ws();
                    if(i>=s.size())bad();
                    if(s[i]==close) {
                        ++i;
                        return out;
                    }
                    if(s[i++]!=',')bad();
                }
            }
            for(auto literal: {
                std::string("true"),std::string("false"),std::string("null")
            })if(s.compare(i,literal.size(),literal)==0) {
                i+=literal.size();
                return literal=="null"?Json():Json(literal=="true");
            }
            size_t start=i;
            bool negative=false;
            if(c=='-') {
                negative=true;
                ++i;
            }
            if(i>=s.size()||s[i]<'0'||s[i]>'9')bad();
            uint64_t magnitude=0,limit=uint64_t(std::numeric_limits<int64_t>::max())+(negative?1:0);
            bool zero=s[i]=='0';
            do {
                unsigned d=s[i++]-'0';
                if(magnitude>(limit-d)/10)throw std::runtime_error("JSON integer overflow");
                magnitude=magnitude*10+d;
                if(zero)break;
            }
            while(i<s.size()&&s[i]>='0'&&s[i]<='9');
            if(i<s.size()&&(s[i]=='.'||s[i]=='e'||s[i]=='E')) {
                if(s[i]=='.') {
                    ++i;
                    size_t digits=i;
                    while(i<s.size()&&s[i]>='0'&&s[i]<='9')++i;
                    if(i==digits)bad();
                }
                if(i<s.size()&&(s[i]=='e'||s[i]=='E')) {
                    ++i;
                    if(i<s.size()&&(s[i]=='+'||s[i]=='-'))++i;
                    size_t digits=i;
                    while(i<s.size()&&s[i]>='0'&&s[i]<='9')++i;
                    if(i==digits)bad();
                }
                Json decimal;
                decimal.kind=Json::Decimal;
                decimal.string=s.substr(start,i-start);
                return decimal;
            }
            (void)start;
            int64_t result=negative?(magnitude==uint64_t(1)<<63?std::numeric_limits<int64_t>::min():-int64_t(magnitude)):int64_t(magnitude);
            return Json(result);
        }
        public:
        explicit Parser(const std::string &input):s(input) {
        }
        Json parse() {
            auto result=value(0);
            ws();
            if(i!=s.size())bad();
            return result;
        }
    };
    inline std::string quote(const std::string &s) {
        std::string out="\"";
        const char *hex="0123456789abcdef";
        for(unsigned char c:s) {
            if(c=='"'||c=='\\') {
                out+='\\';
                out+=char(c);
            }
            else if(c<32) {
                out+="\\u00";
                out+=hex[c>>4];
                out+=hex[c&15];
            }
            else out+=char(c);
        }
        return out+'"';
    }
    inline std::string dump(const Json &j) {
        switch(j.kind) {
            case Json::Null:return "null";
            case Json::Boolean:return j.boolean?"true":"false";
            case Json::Integer:return std::to_string(j.integer);
            case Json::Decimal:return j.string;
            case Json::String:return quote(j.string);
            case Json::Array: {
                std::string out="[";
                for(auto &v:j.array) {
                    if(out.size()>1)out+=',';
                    out+=dump(v);
                }
                return out+"]";
            }
            case Json::Object: {
                std::string out="{";
                for(auto &v:j.object) {
                    if(out.size()>1)out+=',';
                    out+=quote(v.first)+":"+dump(v.second);
                }
                return out+"}";
            }
        }
        throw std::runtime_error("invalid JSON kind");
    }
}
