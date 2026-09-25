#include "../workflow/reduction_execution.h"
#include "../workflow/run_config.h"

SchemeAdditionsReducer::SchemeAdditionsReducer(int count, int schemesCount, int maxFlips, int seed, int blockSize, const std::string &outputPath, int topCount) {
    this->count = count;
    this->schemesCount = schemesCount;
    this->maxFlips = maxFlips;
    this->seed = seed;
    this->blockSize = blockSize;
    this->numBlocks = (count + blockSize - 1) / blockSize;
    this->outputPath = outputPath;
    this->topCount = std::min(topCount, count);

    for (int i = 0; i < 3; i++) {
        this->indices[i].reserve(count);
        this->bestAdditions[i] = 0;

        for (int j = 0; j < count; j++)
            this->indices[i].push_back(j);
    }

    std::cout << "Start memory allocating" << std::endl;
    try {
        metalAllocate(&reducersU, (count + 1) * sizeof(AdditionsReducer<MAX_U_EXPRESSIONS, MAX_U_FRESH_VARIABLES, MAX_U_REAL_VARIABLES, MAX_U_SUBEXPRESSIONS>));
        metalAllocate(&reducersV, (count + 1) * sizeof(AdditionsReducer<MAX_V_EXPRESSIONS, MAX_V_FRESH_VARIABLES, MAX_V_REAL_VARIABLES, MAX_V_SUBEXPRESSIONS>));
        metalAllocate(&reducersW, (count + 1) * sizeof(AdditionsReducer<MAX_W_EXPRESSIONS, MAX_W_FRESH_VARIABLES, MAX_W_REAL_VARIABLES, MAX_W_SUBEXPRESSIONS>));
        metalAllocate(&schemes, (count + 1) * sizeof(SchemeInteger));
        metalAllocate(&states, count * sizeof(RandomState));
    } catch(...) {
        for(void *pointer:{static_cast<void*>(reducersU),static_cast<void*>(reducersV),static_cast<void*>(reducersW),static_cast<void*>(schemes),static_cast<void*>(states)})
            if(pointer) metalFree(pointer);
        throw;
    }
}

bool SchemeAdditionsReducer::read(std::istream &f) {
    if (!(f >> n1 >> n2 >> n3 >> m) || n1 < 1 || n2 < 1 || n3 < 1 || n1 > 16 || n2 > 16 || n3 > 16 || m < 1 || m > MAX_RANK)
        throw std::runtime_error("invalid or unsupported input dimensions and rank");
    std::cout << "Start reading scheme " << n1 << "x" << n2 << "x" << n3 << " with " << m << " multiplications" << std::endl;

    if (m > MAX_U_EXPRESSIONS) {
        std::cout << "Error: expressions count for U ("<< m << ") too big for compiled configuration (" << MAX_U_EXPRESSIONS << ")" << std::endl;
        return false;
    }

    if (m > MAX_V_EXPRESSIONS) {
        std::cout << "Error: expressions count for V ("<< m << ") too big for compiled configuration (" << MAX_V_EXPRESSIONS << ")" << std::endl;
        return false;
    }

    if (n3 * n1 > MAX_W_EXPRESSIONS) {
        std::cout << "Error: expressions count for W ("<< (n3 * n1) << ") too big for compiled configuration (" << MAX_W_EXPRESSIONS << ")" << std::endl;
        return false;
    }

    if (!schemes[0].read(f, n1, n2, n3, m)) {
        std::cout << "Error: readed scheme is invalid" << std::endl;
        return false;
    }

    int realU = schemes[0].getMaxRealVariables(0);
    if (realU > MAX_U_REAL_VARIABLES) {
        std::cout << "Error: real variables for U (" << realU << ") too big for compiled configuration (" << MAX_U_REAL_VARIABLES << ")" << std::endl;
        return false;
    }

    int realV = schemes[0].getMaxRealVariables(1);
    if (realV > MAX_V_REAL_VARIABLES) {
        std::cout << "Error: real variables for V (" << realV << ") too big for compiled configuration (" << MAX_V_REAL_VARIABLES << ")" << std::endl;
        return false;
    }

    int realW = schemes[0].getMaxRealVariables(2);
    if (realW > MAX_W_REAL_VARIABLES) {
        std::cout << "Error: real variables for W (" << realW << ") too big for compiled configuration (" << MAX_W_REAL_VARIABLES << ")" << std::endl;
        return false;
    }

    int subexpressionsU = schemes[0].getMaxSubexpressions(0);
    if (subexpressionsU > MAX_U_SUBEXPRESSIONS) {
        std::cout << "Error: max subexpressions for U (" << subexpressionsU << ") too big for compiled configuration (" << MAX_U_SUBEXPRESSIONS << ")" << std::endl;
        return false;
    }

    int subexpressionsV = schemes[0].getMaxSubexpressions(1);
    if (subexpressionsV > MAX_V_SUBEXPRESSIONS) {
        std::cout << "Error: max subexpressions for V (" << subexpressionsV << ") too big for compiled configuration (" << MAX_V_SUBEXPRESSIONS << ")" << std::endl;
        return false;
    }

    int subexpressionsW = schemes[0].getMaxSubexpressions(2);
    if (subexpressionsW > MAX_W_SUBEXPRESSIONS) {
        std::cout << "Error: max subexpressions for W (" << subexpressionsW << ") too big for compiled configuration (" << MAX_W_SUBEXPRESSIONS << ")" << std::endl;
        return false;
    }

    return true;
}

void SchemeAdditionsReducer::reduce(int maxNoImprovements, int startAdditions) {
    initialize();

    auto startTime = std::chrono::high_resolution_clock::now();
    std::vector<double> elapsedTimes;

    int noImprovements = 0;

    for (int iteration = 1; noImprovements < maxNoImprovements && (metalRounds == 0 || iteration <= metalRounds); iteration++) {
        auto t1 = std::chrono::high_resolution_clock::now();
        reduceIteration(iteration);
        bool improved = updateBest(startAdditions);
        auto t2 = std::chrono::high_resolution_clock::now();

        elapsedTimes.push_back(std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() / 1000.0);
        report(startTime, iteration, elapsedTimes);

        if (improved) {
            noImprovements = 0;
        }
        else {
            noImprovements++;
            std::cout << "No improvements for " << noImprovements << " iterations" << std::endl;
        }
    }
}

SchemeAdditionsReducer::~SchemeAdditionsReducer() {
    if (reducersU) {
        metalFree(reducersU);
        reducersU = nullptr;
    }

    if (reducersV) {
        metalFree(reducersV);
        reducersV = nullptr;
    }

    if (reducersW) {
        metalFree(reducersW);
        reducersW = nullptr;
    }

    if (schemes) {
        metalFree(schemes);
        schemes = nullptr;
    }

    if (states) {
        metalFree(states);
        states = nullptr;
    }
}

void SchemeAdditionsReducer::initialize() {
    metalDispatch("initializeReducersKernel", (numBlocks) * blockSize, blockSize, reducersU, reducersV, reducersW, schemes, states, count, schemesCount, seed);

    bestAdditions[0] = reducersU[count].getAdditions();
    bestAdditions[1] = reducersV[count].getAdditions();
    bestAdditions[2] = reducersW[count].getAdditions();

    bestFreshVars[0] = n1 * n2 * (n1 * n2 - 1);
    bestFreshVars[1] = n2 * n3 * (n2 * n3 - 1);
    bestFreshVars[2] = m * (m - 1);

    reducedAdditions = bestAdditions[0] + bestAdditions[1] + bestAdditions[2];
    reducedFreshVars = bestFreshVars[0] + bestFreshVars[1] + bestFreshVars[2];

    std::cout << "Readed scheme params:" << std::endl;
    std::cout << "- dimensions: " << n1 << "x" << n2 << "x" << n3 << std::endl;
    std::cout << "- multiplications (rank): " << m << std::endl;
    std::cout << "- naive additions (U / V / W / total): " << bestAdditions[0] << " / " << bestAdditions[1] << " / " << bestAdditions[2] << " / " << reducedAdditions << std::endl;
    std::cout << "- max real variables (U / V / W): " << reducersU[count].getMaxRealVariables() << " / " << reducersV[count].getMaxRealVariables() << " / " << reducersW[count].getMaxRealVariables() << std::endl;
    std::cout << "- max unique subexpressions (U / V / W): " << schemes[0].getMaxSubexpressions(0) << " / " << schemes[0].getMaxSubexpressions(1) << " / " << schemes[0].getMaxSubexpressions(2) << std::endl;
    std::cout << std::endl;
}

void SchemeAdditionsReducer::reduceIteration(int iteration) {
    if (maxFlips > 0 && iteration > 1) {
        metalDispatch("flipSchemesKernel", (((schemesCount + blockSize - 1) / blockSize)) * blockSize, blockSize, schemes, states, schemesCount, maxFlips);

    }

    metalDispatch("runReducersKernel", (numBlocks) * blockSize, blockSize, reducersU, reducersV, reducersW, schemes, states, count, schemesCount, maxFlips == 0);

}

bool SchemeAdditionsReducer::updateBestIndependent() {
    std::partial_sort(indices[0].begin(), indices[0].begin() + topCount, indices[0].end(), [this](int index1, int index2) {
        int additions1 = reducersU[index1].getAdditions();
        int additions2 = reducersU[index2].getAdditions();

        if (additions1 != additions2)
            return additions1 < additions2;

        return reducersU[index1].getFreshVars() < reducersU[index2].getFreshVars();
    });

    std::partial_sort(indices[1].begin(), indices[1].begin() + topCount, indices[1].end(), [this](int index1, int index2) {
        int additions1 = reducersV[index1].getAdditions();
        int additions2 = reducersV[index2].getAdditions();

        if (additions1 != additions2)
            return additions1 < additions2;

        return reducersV[index1].getFreshVars() < reducersV[index2].getFreshVars();
    });

    std::partial_sort(indices[2].begin(), indices[2].begin() + topCount, indices[2].end(), [this](int index1, int index2) {
        int additions1 = reducersW[index1].getAdditions();
        int additions2 = reducersW[index2].getAdditions();

        if (additions1 != additions2)
            return additions1 < additions2;

        return reducersW[index1].getFreshVars() < reducersW[index2].getFreshVars();
    });

    int ui = indices[0][0];
    int vi = indices[1][0];
    int wi = indices[2][0];

    int ua = reducersU[ui].getAdditions();
    int va = reducersV[vi].getAdditions();
    int wa = reducersW[wi].getAdditions();

    int uf = reducersU[ui].getFreshVars();
    int vf = reducersV[vi].getFreshVars();
    int wf = reducersW[wi].getFreshVars();

    bool updated = false;

    if (ua < bestAdditions[0] || (ua == bestAdditions[0] && uf < bestFreshVars[0])) {
        bestAdditions[0] = ua;
        bestFreshVars[0] = uf;
        reducersU[count].copyFrom(reducersU[ui]);
        updated = true;
    }

    if (va < bestAdditions[1] || (va == bestAdditions[1] && vf < bestFreshVars[1])) {
        bestAdditions[1] = va;
        bestFreshVars[1] = vf;
        reducersV[count].copyFrom(reducersV[vi]);
        updated = true;
    }

    if (wa < bestAdditions[2] || (wa == bestAdditions[2] && wf < bestFreshVars[2])) {
        bestAdditions[2] = wa;
        bestFreshVars[2] = wf;
        reducersW[count].copyFrom(reducersW[wi]);
        updated = true;
    }

    return updated;
}

bool SchemeAdditionsReducer::updateBestTogether() {
    std::partial_sort(indices[0].begin(), indices[0].begin() + topCount, indices[0].end(), [this](int index1, int index2) {
        int additions1 = reducersU[index1].getAdditions() + reducersV[index1].getAdditions() + reducersW[index1].getAdditions();
        int additions2 = reducersU[index2].getAdditions() + reducersV[index2].getAdditions() + reducersW[index2].getAdditions();

        if (additions1 != additions2)
            return additions1 < additions2;

        int freshVars1 = reducersU[index1].getFreshVars() + reducersV[index1].getFreshVars() + reducersW[index1].getFreshVars();
        int freshVars2 = reducersU[index2].getFreshVars() + reducersV[index2].getFreshVars() + reducersW[index2].getFreshVars();
        return freshVars1 < freshVars2;
    });

    int topIndex = indices[0][0];

    int ua = reducersU[topIndex].getAdditions();
    int va = reducersV[topIndex].getAdditions();
    int wa = reducersW[topIndex].getAdditions();
    int additions = ua + va + wa;

    int uf = reducersU[topIndex].getFreshVars();
    int vf = reducersV[topIndex].getFreshVars();
    int wf = reducersW[topIndex].getFreshVars();
    int freshVars = uf + vf + wf;

    if (additions < reducedAdditions || (additions == reducedAdditions && freshVars < reducedFreshVars)) {
        bestAdditions[0] = ua;
        bestAdditions[1] = va;
        bestAdditions[2] = wa;

        bestFreshVars[0] = uf;
        bestFreshVars[1] = vf;
        bestFreshVars[2] = wf;

        reducersU[count].copyFrom(reducersU[topIndex]);
        reducersV[count].copyFrom(reducersV[topIndex]);
        reducersW[count].copyFrom(reducersW[topIndex]);
        return true;
    }

    return false;
}

bool SchemeAdditionsReducer::updateBest(int startAdditions) {
    bool updated = maxFlips == 0 ? updateBestIndependent() : updateBestTogether();
    int additions = bestAdditions[0] + bestAdditions[1] + bestAdditions[2];
    int freshVars = bestFreshVars[0] + bestFreshVars[1] + bestFreshVars[2];

    if (!updated)
        return false;

    if (additions < reducedAdditions)
        std::cout << "Reduced scheme improved from " << reducedAdditions << " to " << additions << " additions (fresh vars: " << freshVars << ")" << std::endl;
    else
        std::cout << "Reduced scheme improved from " << reducedFreshVars << " fresh vars to " << freshVars << " fresh vars (additions: " << reducedAdditions << ")" << std::endl;

    reducedAdditions = additions;
    reducedFreshVars = freshVars;

    if (reducedAdditions < startAdditions || startAdditions == 0)
        save();

    return true;
}

void SchemeAdditionsReducer::report(std::chrono::high_resolution_clock::time_point startTime, int iteration, const std::vector<double> &elapsedTimes) {
    double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now() - startTime).count() / 1000.0;

    double lastTime = elapsedTimes[elapsedTimes.size() - 1];
    double minTime = *std::min_element(elapsedTimes.begin(), elapsedTimes.end());
    double maxTime = *std::max_element(elapsedTimes.begin(), elapsedTimes.end());
    double meanTime = std::accumulate(elapsedTimes.begin(), elapsedTimes.end(), 0.0) / elapsedTimes.size();

    std::string dimensions = getDimensions();

    std::cout << std::endl << std::left;
    std::cout << "+----------------------------------------------------------------------------------------------------------------------------+" << std::endl;
    std::cout << "| ";
    std::cout << "Size: " << std::setw(24) << dimensions << "   ";
    std::cout << "Reducers: " << std::setw(20) << count << "   ";
    std::cout << "                                 ";
    std::cout << "Iteration: " << std::setw(12) << iteration;
    std::cout << " |" << std::endl;

    std::cout << "| ";
    std::cout << "Rank: " << std::setw(24) << m << "   ";
    std::cout << "Schemes: " << std::setw(21) << schemesCount << "   ";
    std::cout << "                                 ";
    std::cout << "Elapsed: " << std::setw(14) << prettyTime(elapsed);
    std::cout << " |" << std::endl;

    std::cout << std::right;
    std::cout << "+================================+================================+================================+=========================+" << std::endl;
    std::cout << "|           Reducers U           |           Reducers V           |           Reducers W           |          Total          |" << std::endl;
    std::cout << "+------+-------+---------+-------+------+-------+---------+-------+------+-------+---------+-------+-------+---------+-------|" << std::endl;
    std::cout << "| mode | naive | reduced | fresh | mode | naive | reduced | fresh | mode | naive | reduced | fresh | naive | reduced | fresh |" << std::endl;
    std::cout << "+------+-------+---------+-------+------+-------+---------+-------+------+-------+---------+-------+-------+---------+-------+" << std::endl;

    for (int i = 0; i < topCount && i < count; i++) {
        int ui = indices[0][i];
        int vi = indices[maxFlips > 0 ? 0 : 1][i];
        int wi = indices[maxFlips > 0 ? 0 : 2][i];

        std::string modeU = reducersU[ui].getMode();
        std::string modeV = reducersV[vi].getMode();
        std::string modeW = reducersW[wi].getMode();

        int naiveU = reducersU[ui].getNaiveAdditions();
        int naiveV = reducersV[vi].getNaiveAdditions();
        int naiveW = reducersW[wi].getNaiveAdditions();
        int naive = naiveU + naiveV + naiveW;

        int reducedU = reducersU[ui].getAdditions();
        int reducedV = reducersV[vi].getAdditions();
        int reducedW = reducersW[wi].getAdditions();
        int reduced = reducedU + reducedV + reducedW;

        int freshU = reducersU[ui].getFreshVars();
        int freshV = reducersV[vi].getFreshVars();
        int freshW = reducersW[wi].getFreshVars();
        int fresh = freshU + freshV + freshW;

        std::cout << "| ";
        std::cout << std::setw(4) << modeU << "   " << std::setw(5) << naiveU << "   " << std::setw(7) << reducedU << "   " << std::setw(5) << freshU << " | ";
        std::cout << std::setw(4) << modeV << "   " << std::setw(5) << naiveV << "   " << std::setw(7) << reducedV << "   " << std::setw(5) << freshV << " | ";
        std::cout << std::setw(4) << modeW << "   " << std::setw(5) << naiveW << "   " << std::setw(7) << reducedW << "   " << std::setw(5) << freshW << " | ";
        std::cout << std::setw(5) << naive << "   " << std::setw(7) << reduced << "   " << std::setw(5) << fresh << " | ";
        std::cout << std::endl;
    }

    std::cout << "+--------------------------------+--------------------------------+--------------------------------+-------------------------+" << std::endl;
    std::cout << "- iteration time (last / min / max / mean): " << prettyTime(lastTime) << " / " << prettyTime(minTime) << " / " << prettyTime(maxTime) << " / " << prettyTime(meanTime) << std::endl;
    std::cout << "- best additions (U / V / W / total): " << bestAdditions[0] << " / " << bestAdditions[1] << " / " << bestAdditions[2] << " / " << reducedAdditions << std::endl;
    std::cout << "- best fresh vars (U / V / W / total): " << bestFreshVars[0] << " / " << bestFreshVars[1] << " / " << bestFreshVars[2] << " / " << reducedFreshVars << std::endl;
    std::cout << std::endl;
}

void SchemeAdditionsReducer::save() const {
    std::string path = getSavePath();
    int naiveAdditions = reducersU[count].getNaiveAdditions() + reducersV[count].getNaiveAdditions() + reducersW[count].getNaiveAdditions();

    std::ofstream f(path);
    if (!f) throw std::runtime_error("cannot open output: " + path);

    f << "{" << std::endl;
    f << "    \"n\": [" << n1 << ", " << n2 << ", " << n3 << "]," << std::endl;
    f << "    \"m\": " << m << "," << std::endl;
    f << "    \"z2\": false," << std::endl;
    f << "    \"complexity\": {\"naive\": " << naiveAdditions << ", \"reduced\": " << reducedAdditions << "}," << std::endl;
    reducersU[count].write(f, "u", "    ");
    f << "," << std::endl;
    reducersV[count].write(f, "v", "    ");
    f << "," << std::endl;
    reducersW[count].write(f, "w", "    ");
    f << std::endl;
    f << "}" << std::endl;
    f.close();
    if (!f) throw std::runtime_error("cannot write output: " + path);

    std::cout << "Reduced scheme saved to \"" << path << "\"" << std::endl;
}

std::string SchemeAdditionsReducer::getSavePath() const {
    std::stringstream ss;
    ss << outputPath << "/";
    ss << n1 << "x" << n2 << "x" << n3;
    ss << "_m" << m;
    ss << "_cr" << reducedAdditions;
    ss << "_fv" << reducedFreshVars;
    ss << "_cn" << (reducersU[count].getNaiveAdditions() + reducersV[count].getNaiveAdditions() + reducersW[count].getNaiveAdditions());
    ss << "_" << ring;
    ss << "_reduced.json";

    return ss.str();
}

std::string SchemeAdditionsReducer::getDimensions() const {
    std::stringstream ss;
    ss << n1 << "x" << n2 << "x" << n3;
    return ss.str();
}


bool SchemeAdditionsReducer::read(const fgm::SchemeRecord &source) {
    if(source.f2) throw std::runtime_error("signed addition reduction requires ZT");
    std::ostringstream text;
    text<<source.n[0]<<' '<<source.n[1]<<' '<<source.n[2]<<' '<<source.rank<<'\n';
    for(const auto &matrix:source.f) for(const auto &row:matrix) {
        for(auto value:row) text<<value<<' ';
        text<<'\n';
    }
    std::istringstream input(text.str());
    if(!read(input)) return false;
    for(int p=0;p<3;++p) for(int r=0;r<m;++r) for(int c=0;c<schemes[0].nn[p];++c)
        if(schemes[0].uvw[p][r][c]!=source.f[p][r][c]) throw std::runtime_error("admitted effective factors changed during reducer loading");
    return true;
}

fgm::Json SchemeAdditionsReducer::reduceBounded(uint64_t rounds,uint64_t noImprovementLimit,uint64_t targetAdditions,const fgm::AdmissionLimits &limits,const fgm::SchemeRecord &effective) {
    if(!rounds || rounds>INT32_MAX || !noImprovementLimit || noImprovementLimit>INT32_MAX ||
       count<1 || schemesCount<1 || schemesCount>count || maxFlips<0 || maxFlips==INT32_MAX)
        throw std::runtime_error("invalid bounded reduction settings");
    const uint64_t maximumAttempts=fgm::configCheckedMultiply(fgm::configCheckedMultiply(rounds,uint64_t(maxFlips)),uint64_t(schemesCount-1));
    if(maximumAttempts>INT64_MAX) throw fgm::Resource("reduction counters exceed JSON integer capacity");
    struct CounterBuffer {
        uint64_t *data=nullptr;
        explicit CounterBuffer(size_t count) { if(count) metalAllocate(&data,count*sizeof(uint64_t)); }
        ~CounterBuffer() { if(data) metalFree(data); }
    } counters(maxFlips?size_t(schemesCount)*2:0);
    initialize();
    uint64_t completed=0,stagnant=0;
    std::string terminal="round_budget_exhausted";
    if(targetAdditions && uint64_t(reducedAdditions)<=targetAdditions) terminal="addition_target_met";
    else for(uint64_t round=1;round<=rounds;++round) {
        if(maxFlips>0 && round>1)
            metalDispatch("directMutationKernel",size_t((schemesCount+blockSize-1)/blockSize)*blockSize,blockSize,schemes,states,counters.data,schemesCount,maxFlips);
        if(maxFlips>0)
            metalDispatch("runDirectReducersKernel",size_t(numBlocks)*blockSize,blockSize,reducersU,reducersV,reducersW,schemes,states,count,schemesCount);
        else
            metalDispatch("runReducersKernel",size_t(numBlocks)*blockSize,blockSize,reducersU,reducersV,reducersW,schemes,states,count,schemesCount,true);
        bool improved=maxFlips?updateBestTogether():updateBestIndependent();
        reducedAdditions=bestAdditions[0]+bestAdditions[1]+bestAdditions[2];
        reducedFreshVars=bestFreshVars[0]+bestFreshVars[1]+bestFreshVars[2];
        completed=round; stagnant=improved?0:stagnant+1;
        std::cout<<"Native reduction progress: round "<<round<<" best_additions "<<reducedAdditions<<" stagnant "<<stagnant<<std::endl;
        if(targetAdditions && uint64_t(reducedAdditions)<=targetAdditions) { terminal="addition_target_met"; break; }
        if(stagnant>=noImprovementLimit) { terminal="no_improvement_limit"; break; }
    }
    if(!reducersU[count].isValid() || !reducersV[count].isValid() || !reducersW[count].isValid()) throw std::runtime_error("invalid retained best reducer state");
    fgm::ReductionRecordBuffer buffer(limits.record);
    std::ostream encoded(&buffer);
    encoded.exceptions(std::ios::badbit | std::ios::failbit);
    encoded<<'{'; reducersU[count].write(encoded,"u",""); encoded<<',';
    reducersV[count].write(encoded,"v",""); encoded<<','; reducersW[count].write(encoded,"w",""); encoded<<'}';
    fgm::Json circuit=fgm::Parser(buffer.str()).parse();
    auto dimensions=fgm::Json::list(); for(int n:{n1,n2,n3}) dimensions.array.emplace_back(int64_t(n));
    circuit.object["n"]=dimensions;
    // Each retained U output is one multiplication. The best may precede the
    // final walker and may have a different rank from the imported scheme.
    circuit.object["m"]=fgm::Json(int64_t(circuit.at("u").array.size()));
    circuit.object["z2"]=fgm::Json(false);
    auto complexity=fgm::Json::dict();
    complexity.object["naive"]=fgm::Json(int64_t(reducersU[count].getNaiveAdditions()+reducersV[count].getNaiveAdditions()+reducersW[count].getNaiveAdditions()));
    complexity.object["reduced"]=fgm::Json(int64_t(reducedAdditions)); circuit.object["complexity"]=complexity;
    fgm::AdmissionContext verifier(limits);
    auto reconstructed=fgm::verifyReductionCircuit(circuit,effective,limits,maxFlips==0);
    circuit.object["scheme_id"]=fgm::Json(verifier.identity(reconstructed,true));
    circuit.object["factors_id"]=fgm::Json(verifier.identity(reconstructed,false));
    uint64_t attempted=0,applied=0;
    if(counters.data) for(int i=0;i<schemesCount;++i) {
        attempted=fgm::configCheckedAdd(attempted,counters.data[2*i]);
        applied=fgm::configCheckedAdd(applied,counters.data[2*i+1]);
    }
    if(attempted>maximumAttempts || applied>attempted) throw std::runtime_error("invalid mutation accounting");
    auto result=fgm::Json::dict(); result.object["circuit"]=std::move(circuit);
    result.object["result_scheme_id"]=fgm::Json(verifier.identity(reconstructed,true));
    result.object["result_factors_id"]=fgm::Json(verifier.identity(reconstructed,false));
    result.object["result_rank"]=fgm::Json(int64_t(reconstructed.rank));
    result.object["rounds_completed"]=fgm::Json(int64_t(completed));
    result.object["flip_attempts"]=fgm::Json(int64_t(attempted)); result.object["flips_applied"]=fgm::Json(int64_t(applied));
    result.object["verified_circuit_additions"]=fgm::Json(int64_t(reconstructed.operations));
    result.object["terminal_reason"]=fgm::Json(terminal);
    result.object["verification"]=fgm::Json("exact-Z circuit reconstruction and tensor");
    return result;
}
