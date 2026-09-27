// Host-only verification of the production best-circuit binding boundary.
#include "../../src/metal/host.h"
#include "../../src/workflow/reduction_execution.h"
// Shared by the existing host and GPU test drivers; never used in production.
fgm::Json constructorTest(const fgm::Json &request,bool gpu=false) {
    fgm::Matrix targets;
    for(const auto &row:request.at("targets").array) {
        std::vector<int64_t> values; for(const auto &x:row.array) values.push_back(x.num()); targets.push_back(values);
    }
    auto p=fgm_constructor::prepare(targets,int(request.at("inputs").num()));
    auto response=fgm::Json::dict(), results=fgm::Json::list();
    response.object["directions"]=fgm::Json(int64_t(p.directions));
    response.object["candidate_slots"]=fgm::Json(int64_t(p.candidateCount));
#ifdef FGM_CONSTRUCTOR_GPU_TEST
    fgm::ReductionBuffer<fgm_constructor::Problem> problem(1); *problem.data=p;
    fgm::ReductionBuffer<fgm_constructor::Witness> witnesses(128);
#endif
    for(int candidate=-1;candidate<p.candidateCount;++candidate) {
        fgm_constructor::Witness w{};
#ifdef FGM_CONSTRUCTOR_GPU_TEST
        if(gpu) {
            int offset=(candidate+1)%128;
            if(!offset) {
                int count=std::min(128,p.candidateCount-candidate);
                metalDispatch("constructorClosureKernel",size_t((count+31)/32)*32,32,problem.data,witnesses.data,candidate,count);
            }
            w=witnesses.data[offset];
        } else
#endif
        fgm_constructor::close(p,candidate,w);
        if(request.has("tamper") && w.status==1 && w.count) {
            auto kind=request.at("tamper").str();
            if(kind=="dependency") w.gates[0].left=w.gates[0].out;
            else if(kind=="sign") w.gates[0].leftSign=0;
            else if(kind=="capacity") w.count=25;
            else if(kind=="availability") w.available^=uint64_t(1)<<32;
            else if(kind=="auxiliary") for(int i=0;i<w.count;++i) if(w.gates[i].out==p.directions) {
                std::swap(w.gates[i].left,w.gates[i].right); std::swap(w.gates[i].leftSign,w.gates[i].rightSign);
            }
        }
        auto stage=fgm::constructorWitness(p,w,candidate);
        auto result=fgm::Json::dict(), gates=fgm::Json::list();
        if(w.status==1) { result.object["outputs"]=stage.outputs; result.object["fresh"]=stage.fresh; }
        result.object["status"]=fgm::Json(int64_t(w.status));
        result.object["available"]=fgm::Json(int64_t(w.available));
        result.object["count"]=fgm::Json(int64_t(w.count));
        for(int i=0;i<w.count;++i) {
            const auto &g=w.gates[i]; auto gate=fgm::Json::list();
            for(int v:{g.out,g.left,g.right,g.leftSign,g.rightSign}) gate.array.emplace_back(int64_t(v));
            gates.array.push_back(gate);
        }
        result.object["gates"]=gates; results.array.push_back(result);
    }
    response.object["results"]=results; return response;
}
#ifndef FGM_CONSTRUCTOR_GPU_TEST
int main() {
    try {
        std::string bytes; char byte;
        while(std::cin.get(byte)) { if(bytes.size()==1048576) throw fgm::Resource("test input bound"); bytes+=byte; }
        auto request=fgm::Parser(bytes).parse();
        if(request.has("targets")) {
            std::cout<<fgm::dump(constructorTest(request))<<'\n'; return 0;
        }
        if(request.has("transpose")) {
            fgm::CircuitStage source{request.at("outputs"),request.at("fresh")};
            fgm::AdmissionContext context; uint64_t before=0,after=0;
            auto original=context.reconstructStage(source.outputs,source.fresh,uint64_t(request.at("inputs").num()),before);
            auto stage=fgm::transposeStage(source,int(request.at("inputs").num()));
            auto transposed=context.reconstructStage(stage.outputs,stage.fresh,source.outputs.array.size(),after);
            for(size_t r=0;r<original.size();++r) for(size_t c=0;c<original[r].size();++c)
                if(original[r][c]!=transposed[c][r]) throw std::runtime_error("incorrect transposition");
            auto response=fgm::Json::dict(); response.object["outputs"]=stage.outputs; response.object["fresh"]=stage.fresh;
            response.object["count"]=fgm::Json(int64_t(after)); std::cout<<fgm::dump(response)<<'\n'; return 0;
        }
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
        auto stages=fgm::Json::dict();
        for(int p=0;p<3;++p)stages.object[std::string(1,"uvw"[p])]=fgm::Json(int64_t(result.operationsByStage[p]));
        response.object["verified_circuit_additions_by_stage"]=std::move(stages);
        std::cout<<fgm::dump(response)<<'\n';
        return 0;
    } catch(const fgm::Resource &error) { std::cerr<<error.what()<<'\n'; return 2; }
      catch(const std::exception &error) { std::cerr<<error.what()<<'\n'; return 1; }
}

#endif
