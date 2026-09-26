#include "pool.h"
#include <iostream>
using namespace fgm;
void require(bool value,const char *message){if(!value)throw std::runtime_error(message);}
template<class F>void rejects(F f,const char *message){bool failed=false;try{f();}catch(const std::exception&){failed=true;}require(failed,message);}
PoolMember structural(const std::string &id,uint32_t rank,uint64_t weight=0){PoolMember m;m.id=id;m.scheme.rank=rank;m.weight=weight;return m;}
PoolMember verified(unsigned zeroFactor,AdmissionContext &context){
    SchemeRecord s;s.n={1,1,1};s.rank=2;for(auto &factor:s.f)factor={{1},{1}};s.f[zeroFactor][1][0]=0;
    require(context.verify(s),"invalid structural reserve fixture");
    return {context.identity(s,true),s,uint64_t(context.analyze(s).at("potential_pairs").num())};
}
int main(int argc,char **argv){try{
    require(argc>=2,"pool test command required");std::string command=argv[1];
    if(command=="selection"){
        require(argc>=6,"selection mode seed count weights required");PoolSettings settings;settings.selector=argv[2];settings.capacity=uint64_t(argc-5);settings.reserve=settings.capacity;
        RankPools pools(settings);for(int i=5;i<argc;++i)pools.admit(structural(std::to_string(i-5),2,std::stoull(argv[i])));
        std::mt19937_64 rng(std::stoull(argv[3]));for(unsigned i=0;i<std::stoul(argv[4]);++i){auto member=pools.select(2,rng);require(member,"missing parent");std::cout<<member->id<<'\n';}std::cout<<"next "<<rng()<<'\n';
    }else if(command=="fifo"){
        RankPools pools(PoolSettings{2,2,1024*1024,1,"uniform"});pools.admit(structural("a",2));pools.admit(structural("b",2));require(!pools.admit(structural("a",2)),"duplicate admitted");pools.admit(structural("c",2));std::mt19937_64 rng(7);for(int i=0;i<32;++i)std::cout<<pools.select(2,rng)->id<<'\n';std::cout<<"next "<<rng()<<'\n';
    }else if(command=="stages"){
        RankPools pools(PoolSettings{3,2,1024*1024,2,"uniform"});pools.admit(structural("a",5));pools.admit(structural("b",3));require(pools.nextStage(8)==8,"smaller-population fallback used");pools.admit(structural("c",5));require(pools.nextStage(8)==5,"eligible rank not selected");pools.admit(structural("d",3));require(pools.nextStage(8)==3,"lowest eligible rank not selected");require(pools.nextStage(3)==3,"stage moved upward");std::cout<<"PASS stages\n";
    }else if(command=="reserves"){
        AdmissionContext context;auto a=verified(0,context),b=verified(1,context);PoolSettings settings{1,1,1024*1024,1,"uniform"};RankPools pools(settings);pools.admit(a);pools.stageEntry(2);pools.admit(b);pools.refill(2);std::mt19937_64 rng(0);require(pools.select(2,rng)->id==b.id,"reserve replaced nonempty roster");auto state=pools.snapshot(context);state.object["active"]=Json::list();RankPools restored(settings);restored.restore(state,context);require(restored.select(2,rng)->id==a.id,"empty roster not refilled from stage reserve");require(restored.select(3,rng)==nullptr,"missing reserve fabricated parent");std::cout<<"PASS reserves\n";
    }else if(command=="limits"){
        rejects([]{RankPools p(PoolSettings{1,1,1024*1024,2,"uniform"});},"threshold exceeds capacity accepted");RankPools pools(PoolSettings{2,1,128,1,"uniform"});rejects([&]{pools.admit(structural("a",2));},"total memory overflow accepted");RankPools weighted(PoolSettings{2,1,1024*1024,1,"flips"});weighted.admit(structural("a",2,UINT64_MAX));weighted.admit(structural("b",2,1));std::mt19937_64 rng(0),untouched(0);rejects([&]{weighted.select(2,rng);},"weight sum overflow accepted");require(rng()==untouched(),"overflow consumed RNG");std::cout<<"PASS limits\n";
    }else throw std::runtime_error("unknown pool test command");
    return 0;
}catch(const std::exception &e){std::cerr<<e.what()<<'\n';return 1;}}
