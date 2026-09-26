#include "journal.h"
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <fcntl.h>
#include <unistd.h>
using namespace fgm;
namespace fs=std::filesystem;
void check(bool value,const char *why){if(!value)throw std::runtime_error(why);}
template<class F>void rejects(F f,const char *why){bool failed=false;try{f();}catch(const std::exception&){failed=true;}check(failed,why);}
std::string id(int n){return "fgm-scheme-v1:"+sha256Bytes(std::to_string(n));}
Json transaction(int n,const std::string &origin="discovery"){
    Json j=Json::dict(),a=Json::dict();a.object["scheme_id"]=Json(id(n));a.object["origin"]=Json(origin);a.object["rank"]=Json(int64_t(23));a.object["domain"]=Json("ZT");j.object["admissions"]=Json::list();j.object["admissions"].array.push_back(a);j.object["pool_snapshot"]=Json("controller-owned");return j;
}
CommitReceipt append(Journal &j,const Json &t){return j.append(t,j.reserve(32768));}
std::string bytes(const fs::path &p){std::ifstream in(p,std::ios::binary);return std::string((std::istreambuf_iterator<char>(in)),{});}
void write(const fs::path &p,const std::string &data){std::ofstream out(p,std::ios::binary|std::ios::trunc);out<<data;}
std::map<std::string,std::string> inventory(const fs::path &path){std::map<std::string,std::string> files;for(const auto &entry:fs::directory_iterator(path))files[entry.path().filename().string()]=bytes(entry.path());return files;}
int main(int argc,char **argv){
    try{
        check(argc==3,"scenario and output directory required");std::string scenario=argv[1];fs::path path=argv[2];JournalLimits limits{4*1024*1024,32768,1024};
        if(scenario=="fixture"){
            std::string text((std::istreambuf_iterator<char>(std::cin)),{});auto transactions=Parser(text).parse();
            check(transactions.kind==Json::Array,"fixture transaction array required");Journal journal(path,limits,true);
            for(const auto &tx:transactions.array)append(journal,tx);
        }else if(scenario=="basic"){
            {Journal journal(path,limits,true);auto first=append(journal,transaction(1,"import"));check(first.novelIds.size()==1&&first.creditedIds.empty(),"import credit");check(journal.contains(id(1)),"import membership");check(append(journal,transaction(1)).creditedIds.empty(),"import rediscovery credit");check(append(journal,transaction(2)).creditedIds.size()==1,"new discovery credit");rejects([&]{Journal second(path,limits);},"concurrent writer accepted");}
            Journal recovered(path,limits);size_t count=0,credits=0;recovered.recover([&](const Json &j,const CommitReceipt&r){++count;credits+=r.creditedIds.size();check(j.at("pool_snapshot").str()=="controller-owned","controller payload lost");check(j.at("admissions").array[0].at("rank").num()==23,"rank lost");});check(count==3&&credits==1,"replay accounting");check(recovered.contains(id(1))&&recovered.contains(id(2))&&!recovered.contains(id(3)),"recovered membership");
            auto before=bytes(path/"journal.bin");size_t replayed=0;auto result=Journal::replayReadOnly(path,limits,[&](const Json&,const CommitReceipt&){++replayed;});check(result.sequence==3&&replayed==3&&before==bytes(path/"journal.bin"),"read-only replay mismatch");
        }else if(scenario=="short-writes"){
            Journal journal(path,limits,true);Journal::testFaults(-1,-1,7);check(append(journal,transaction(1)).creditedIds.size()==1,"short writes lost commit");Journal::testFaults();
        }else if(scenario=="torn-tail"){
            {Journal journal(path,limits,true);append(journal,transaction(1,"import"));Journal::testFaults(100);rejects([&]{append(journal,transaction(2));},"write injection ignored");Journal::testFaults();rejects([&]{journal.contains(id(1));},"failed writer not poisoned");}
            auto original=bytes(path/"journal.bin");auto inspection=Journal::replayReadOnly(path,limits);check(inspection.sequence==1&&inspection.incompleteTailBytes==100,"read-only tail description");check(bytes(path/"journal.bin")==original,"read-only trimmed tail");
            Journal recovered(path,limits);check(recovered.state().sequence==1,"torn tx committed");check(recovered.state().retainedTails.size()==1,"missing tail evidence");check(bytes(recovered.state().retainedTails[0])==original.substr(original.size()-100),"tail bytes changed");check(!recovered.contains(id(2)),"torn discovery indexed");check(append(recovered,transaction(2)).creditedIds.size()==1,"post-tail discovery lost");
        }else if(scenario=="sync-failure"||scenario=="index-sync-failure"||scenario=="recovery-sync-failure"){
            std::string floor;
            {Journal journal(path,limits,true);floor=bytes(path/"committed-head.json");Journal::testFaults(-1,scenario=="index-sync-failure"?1:0);rejects([&]{append(journal,transaction(7));},"sync injection ignored");Journal::testFaults();}
            check(bytes(path/"committed-head.json")==floor,"failed append advanced head");
            check(!Journal::replayReadOnly(path,limits).derivedIndexCurrent,"stale index reported current");
            const auto journalPath=fs::canonical(path/"journal.bin");
            if(scenario=="recovery-sync-failure"){
                Journal::testIoReset(journalPath);
                rejects([&]{Journal recovered(path,limits);},"recovery journal sync injection ignored");
                check(bytes(path/"committed-head.json")==floor,"failed recovery journal sync advanced head");
                bool failedJournalSync=false;
                for(const auto &event:Journal::testIoEvents()){
                    check(event.operation!="publish","failed recovery published head");
                    if(event.operation=="sync"&&event.path==journalPath){check(!event.succeeded,"targeted journal sync unexpectedly succeeded");failedJournalSync=true;}
                }
                check(failedJournalSync,"recovery did not attempt authoritative journal sync");
            }
            Journal::testIoReset();
            Journal recovered(path,limits);
            bool journalSynced=false,published=false;
            for(const auto &event:Journal::testIoEvents()){
                if(event.operation=="sync"&&event.path==journalPath&&event.succeeded)journalSynced=true;
                if(event.operation=="publish"){
                    check(event.path.filename()=="committed-head.json","unexpected publication target");
                    check(journalSynced,"recovery published head before authoritative journal sync");published=true;
                }
            }
            check(journalSynced&&published,"recovery sync/publication evidence missing");
            check(bytes(path/"committed-head.json")!=floor,"recovery did not advance head");
            check(recovered.state().sequence==1&&recovered.contains(id(7)),"complete unacknowledged frame lost");
            size_t credits=0;Journal::replayReadOnly(path,limits,[&](const Json&,const CommitReceipt &receipt){credits+=receipt.creditedIds.size();});
            check(credits==1,"recovered discovery credit mismatch");
            // Check recovery on its own before a later append can synchronize
            // journal.bin and mask the missing recovery durability barrier.
            Journal::testIoReset();
            if(scenario!="recovery-sync-failure")check(append(recovered,transaction(7)).creditedIds.empty(),"unacknowledged frame credited twice");
        }else if(scenario=="head-sync-failure"){
            std::string floor;
            {Journal journal(path,limits,true);floor=bytes(path/"committed-head.json");Journal::testFaults(-1,3);rejects([&]{append(journal,transaction(7));},"head sync injection ignored");Journal::testFaults();}
            check(bytes(path/"committed-head.json")==floor,"failed head sync advanced floor");
            auto prior=inventory(path);size_t pending=0;for(const auto &entry:prior)if(entry.first.rfind("head-pending-",0)==0)++pending;check(pending==1,"head temporary evidence missing");
            check(Journal::replayReadOnly(path,limits).sequence==1,"complete beyond-floor commit not replayed");check(inventory(path)==prior,"read-only recovery mutated head evidence");
            Journal recovered(path,limits);check(recovered.contains(id(7)),"beyond-floor discovery lost");for(const auto &entry:prior)if(entry.first.rfind("head-pending-",0)==0)check(bytes(path/entry.first)==entry.second,"orphan head evidence changed");check(append(recovered,transaction(7)).creditedIds.empty(),"head failure rediscovery credited");
        }else if(scenario=="head-required"){
            std::string genesis,first,full,head;
            {Journal journal(path,limits,true);genesis=bytes(path/"committed-head.json");append(journal,transaction(1));first=bytes(path/"journal.bin");append(journal,transaction(2));full=bytes(path/"journal.bin");head=bytes(path/"committed-head.json");}
            auto rejectUnchanged=[&]{auto prior=inventory(path);rejects([&]{Journal::replayReadOnly(path,limits);},"invalid head accepted read-only");rejects([&]{Journal recovered(path,limits);},"invalid head accepted writable");check(inventory(path)==prior,"invalid history mutated during recovery");};
            write(path/"journal.bin",first);rejectUnchanged();
            fs::remove(path/"index.ids");fs::remove(path/"index.json");rejectUnchanged();
            write(path/"journal.bin",full);fs::remove(path/"committed-head.json");rejectUnchanged();
            write(path/"committed-head.json","corrupt");rejectUnchanged();
            auto changed=Parser(head).parse();changed.object["frame_sha256"]=Json(sha256Bytes("wrong head"));changed.object.erase("checksum_sha256");changed.object["checksum_sha256"]=Json(sha256Bytes(dump(changed)));write(path/"committed-head.json",dump(changed));rejectUnchanged();
            changed=Parser(head).parse();changed.object["committed_bytes"]=Json(changed.at("committed_bytes").num()+1);changed.object.erase("checksum_sha256");changed.object["checksum_sha256"]=Json(sha256Bytes(dump(changed)));write(path/"committed-head.json",dump(changed));rejectUnchanged();
            // A stale valid floor allows durable complete transactions beyond it.
            write(path/"committed-head.json",genesis);auto prior=inventory(path);auto read=Journal::replayReadOnly(path,limits);check(read.sequence==2&&!read.derivedIndexCurrent,"stale floor with missing index failed");check(prior==inventory(path),"stale floor inspection mutated files");
            Journal recovered(path,limits);check(recovered.state().sequence==2&&recovered.contains(id(1))&&recovered.contains(id(2)),"stale floor recovery lost history");check(bytes(path/"committed-head.json")==head,"recovery did not advance durable floor");
        }else if(scenario=="corruption"){
            {Journal journal(path,limits,true);append(journal,transaction(1));}
            auto original=bytes(path/"journal.bin");original[180]^=1;write(path/"journal.bin",original);rejects([&]{Journal recovered(path,limits);},"committed corruption recovered");check(bytes(path/"journal.bin")==original,"corrupt committed frame changed");
        }else if(scenario=="rebuild"){
            {Journal journal(path,limits,true);Json tx=transaction(0,"import");for(int i=1;i<80;++i)tx.object["admissions"].array.push_back(transaction(i,i%2?"import":"discovery").at("admissions").array[0]);auto r=append(journal,tx);check(r.novelIds.size()==80&&r.creditedIds.size()==39,"batch accounting");}
            fs::remove(path/"index.ids");fs::remove(path/"index.json");{Journal recovered(path,limits);for(int i=0;i<80;++i)check(recovered.contains(id(i)),"external rebuild lost identity");check(!recovered.contains(id(80)),"external rebuild false identity");}
            auto index=bytes(path/"index.ids");auto forged=id(999).substr(std::string("fgm-scheme-v1:").size())+'\n';write(path/"index.ids",forged);auto meta=Parser(bytes(path/"index.json")).parse();meta.object["count"]=Json(int64_t(1));meta.object["index_sha256"]=Json(sha256Bytes(forged));write(path/"index.json",dump(meta));
            check(!Journal::replayReadOnly(path,limits).derivedIndexCurrent,"rehashed fabricated index accepted");
            write(path/"index.ids","corrupt cache");Journal recovered(path,limits);check(recovered.contains(id(79)),"corrupt cache not rebuilt");check(append(recovered,transaction(1)).creditedIds.empty(),"historical import credited after rebuild");
        }else if(scenario=="storage"){
            Journal journal(path,JournalLimits{4096,32768,1024},true);auto before=bytes(path/"journal.bin");rejects([&]{journal.reserve(32768);},"storage over-reserved");check(bytes(path/"journal.bin")==before,"failed reservation mutated journal");
        }else if(scenario=="invalid"){
            Journal journal(path,limits,true);Json tx=transaction(1);tx.object["admissions"].array.push_back(tx.at("admissions").array[0]);rejects([&]{append(journal,tx);},"duplicate tx identity accepted");tx=transaction(1);tx.object["_journal"]=Json::dict();rejects([&]{append(journal,tx);},"supplied accounting accepted");auto stale=journal.reserve(16384);append(journal,transaction(1));rejects([&]{journal.append(transaction(2),stale);},"stale reservation accepted");check(journal.state().sequence==1,"rejected append changed state");
        }else throw std::runtime_error("unknown scenario");
        std::cout<<"PASS "<<scenario<<'\n';return 0;
    }catch(const std::exception&e){Journal::testFaults();std::cerr<<e.what()<<'\n';return 1;}
}
