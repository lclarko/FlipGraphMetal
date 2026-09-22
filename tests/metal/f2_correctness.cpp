#include "../../src/metal/host.h"

int main() {
    try {
        std::filesystem::create_directories("build/metal/f2-test-output");
        SchemeZ2 *gpu, *other;
        RandomState *states;
        metalAllocate(&gpu, sizeof(SchemeZ2));
        metalAllocate(&other, sizeof(SchemeZ2));
        metalAllocate(&states, sizeof(RandomState));
        for (int operation = 0; operation <= 12; operation++) {
            SchemeZ2 cpu;
            std::ifstream fixture("tests/metal/fixtures/strassen_3x3_f2.txt");
            if (!cpu.read(fixture)) throw std::runtime_error("invalid F2 fixture");
            RandomState state = {7713};
            for (int i = 0; i < 100; i++) cpu.tryFlip(state);
            *gpu = cpu;
            *states = state;
            other->initializeNaive(operation == 10 ? 2 : 3, operation == 10 ? 2 : 3, operation == 10 ? 2 : 3);
            switch (operation) {
                case 0: for (int j = 0; j < 32; j++) cpu.tryFlip(state); break;
                case 1: cpu.tryPlus(state); break;
                case 2: cpu.tryExpand(2,state); break;
                case 3: cpu.tryReduce(); break;
                case 4: cpu.tryProject(state); break;
                case 5: cpu.tryExtend(state); break;
                case 6: cpu.tryProduct(state); break;
                case 7: cpu.swapBasis(state); break;
                case 8: cpu.swapSize(state); break;
                case 9: cpu.tryMerge(*other,state); break;
                case 10: cpu.tryProduct(*other); break;
                case 11: cpu.sandwiching(state); break;
                case 12: cpu.tryReduceGauss(state); break;
            }
            if (!cpu.validate()) throw std::runtime_error("invalid CPU F2 transformation " + std::to_string(operation));
            metalDispatch("transformationKernel",1,1,gpu,states,other,operation);
            if (cpu.m != gpu->m || state.value != states->value) throw std::runtime_error("F2 rank or RNG mismatch");
            for (int p = 0; p < 3; p++) {
                if (cpu.n[p] != gpu->n[p] || cpu.nn[p] != gpu->nn[p] || cpu.flips[p].size != gpu->flips[p].size)
                    throw std::runtime_error("F2 shape mismatch");
                for (int r = 0; r < cpu.m; r++)
                    if (cpu.uvw[p][r] != gpu->uvw[p][r]) throw std::runtime_error("F2 coefficient mismatch");
                for (size_t i = 0; i < cpu.flips[p].size; i++)
                    if (cpu.flips[p].pairs[i] != gpu->flips[p].pairs[i]) throw std::runtime_error("F2 flip ordering mismatch");
            }
            gpu->save("build/metal/f2-test-output/transform-" + std::to_string(operation) + ".json");
        }
        metalFree(gpu); metalFree(other); metalFree(states);
        std::cout << "PASS: 13 F2 CPU/Metal transformations and random states\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
