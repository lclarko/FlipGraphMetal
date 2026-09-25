// Host-only verification of the production best-circuit binding boundary.
#include "host.h"
#include "reduction_execution.h"
int main() {
    try {
        std::string bytes; char byte;
        while(std::cin.get(byte)) { if(bytes.size()==1048576) throw fgm::Resource("test input bound"); bytes+=byte; }
        auto request=fgm::Parser(bytes).parse();
        if(request.has("serialize")) {
            fgm::ReductionRecordBuffer buffer(uint64_t(request.at("limit").num()));
            std::ostream stream(&buffer);stream.exceptions(std::ios::badbit|std::ios::failbit);
            stream<<request.at("serialize").str();
            auto response=fgm::Json::dict();response.object["bytes"]=fgm::Json(buffer.str());
            std::cout<<fgm::dump(response)<<'\n';return 0;
        }
        fgm::AdmissionLimits limits;
        if(request.has("work")) limits.work=uint64_t(request.at("work").num());
        fgm::AdmissionContext context(limits);
        auto effective=context.fromJson(request.at("effective"),"ZT");
        if(!context.verify(effective)) throw std::runtime_error("invalid effective input");
        auto result=fgm::verifyReductionCircuit(request.at("circuit"),effective,limits,request.at("fixed").boolean);
        auto response=context.schemeJson(result);
        response.object["factors_id"]=fgm::Json(context.identity(result,false));
        response.object["verified_circuit_additions"]=fgm::Json(int64_t(result.operations));
        std::cout<<fgm::dump(response)<<'\n';
        return 0;
    } catch(const fgm::Resource &error) { std::cerr<<error.what()<<'\n'; return 2; }
      catch(const std::exception &error) { std::cerr<<error.what()<<'\n'; return 1; }
}
