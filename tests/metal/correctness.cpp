#include "../../src/metal/host.h"
#include <cstdlib>
#include "candidate_capacity.h"

void require(bool condition, const std::string &message) {
    if (!condition) throw std::runtime_error(message);
}

bool equal(const SchemeInteger &a, const SchemeInteger &b) {
    if (a.m != b.m) return false;
    for (int p = 0; p < 3; p++) {
        if (a.n[p] != b.n[p] || a.nn[p] != b.nn[p] || a.flips[p].size != b.flips[p].size) return false;
        for (int r = 0; r < a.m; r++)
            if (a.uvw[p][r] != b.uvw[p][r] || a.uvw[p][r].valid != b.uvw[p][r].valid) return false;
        for (size_t i = 0; i < a.flips[p].size; i++)
            if (a.flips[p].pairs[i] != b.flips[p].pairs[i]) return false;
    }
    return true;
}

void textScheme(std::ostream &out, const SchemeInteger &scheme) {
    for (int p = 0; p < 3; p++)
        for (int r = 0; r < scheme.m; r++) {
            for (int i = 0; i < scheme.nn[p]; i++) out << scheme.uvw[p][r][i] << ' ';
            out << '\n';
        }
}

int main(int argc, char **argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--help") {
            std::cout << "Usage: " << argv[0] << " --output-dir NEW_DIRECTORY\n";
            return 0;
        }
        if (argc != 3 || std::string(argv[1]) != "--output-dir" || !argv[2][0] || std::string(argv[2]).rfind("--", 0) == 0)
            throw std::runtime_error("expected --output-dir NEW_DIRECTORY; parent directory must exist");
        const std::filesystem::path output(argv[2]);
        if (!std::filesystem::create_directory(output))
            throw std::runtime_error("output directory already exists: " + output.string());
        checkCandidateCapacityAndLayout();
        using U = AdditionsReducer<350,250,32,2016>;
        using W = AdditionsReducer<64,500,175,61075>;
        int *arithmetic;
        metalAllocate(&arithmetic, 64 * 9 * 4 * sizeof(int));
        metalDispatch("arithmeticKernel", 64 * 9, 32, arithmetic);
        for (int bit = 0; bit < 64; bit++) {
            for (int b = -1; b <= 1; b++) {
                for (int a = -1; a <= 1; a++) {
                    int index = 4 * (bit * 9 + (b + 1) * 3 + a + 1);
                    require(arithmetic[index] == (std::abs(a+b) <= 1), "sum validity mismatch");
                    require(arithmetic[index+2] == (std::abs(a-b) <= 1), "difference validity mismatch");
                    if (std::abs(a+b) <= 1) require(arithmetic[index+1] == a+b, "sum coefficient mismatch");
                    if (std::abs(a-b) <= 1) require(arithmetic[index+3] == a-b, "difference coefficient mismatch");
                }
            }
        }
        metalFree(arithmetic);
        Addition *randomAdditions;
        RandomState *randomStates;
        metalAllocate(&randomAdditions, 64 * sizeof(Addition));
        metalAllocate(&randomStates, 64 * sizeof(RandomState));
        for (int i = 0; i < 64; i++) randomStates[i] = {uint32_t(77 + i)};
        metalDispatch("randomAdditionKernel", 64, 4, randomAdditions, randomStates);
        for (int i = 0; i < 64; i++) {
            Addition expected(i + 1);
            RandomState state = {uint32_t(77 + i)};
            expected.random(state);
            require(expected == randomAdditions[i] && state.value == randomStates[i].value, "random addition mismatch");
            if (i < 63) require((expected.values >> (i + 1)) == 0, "random bits outside declared width");
        }
        metalFree(randomAdditions); metalFree(randomStates);
        SchemeInteger initial;
        std::ifstream fixture("tests/metal/fixtures/strassen_3x3.txt");
        require(initial.read(fixture), "invalid signed-ternary fixture");
        RandomState preparation = {1234567};
        for (int i = 0; i < 100; i++) initial.tryFlip(preparation);
        require(initial.validate(), "CPU preparation tensor invalid");
        initial.save((output / "input.json").string());
        {
            std::ofstream out((output / "input.txt"));
            out << "3 3 3 " << initial.m << '\n';
            textScheme(out, initial);
        }
        {
            std::ofstream out((output / "minimizer.txt"));
            out << "3 3 3 " << initial.m << " 1\n";
            textScheme(out, initial);
        }
        SchemeInteger *gpu, *other;
        RandomState *states;
        metalAllocate(&gpu, 4 * sizeof(SchemeInteger));
        metalAllocate(&other, sizeof(SchemeInteger));
        metalAllocate(&states, 4 * sizeof(RandomState));
        for (int operation = 0; operation <= 10; operation++) {
            SchemeInteger cpu[4]; RandomState expected[4];
            other->initializeNaive(operation == 10 ? 2 : 3, operation == 10 ? 2 : 3, operation == 10 ? 2 : 3);
            for (int i = 0; i < 4; i++) {
                initial.copyTo(cpu[i]);
                if (operation == 3) {
                    int r = cpu[i].m++;
                    int q = 0;
                    while (cpu[i].uvw[2][0][q] != 0) q++;
                    cpu[i].uvw[0][r] = cpu[i].uvw[0][0];
                    cpu[i].uvw[1][r] = cpu[i].uvw[1][0];
                    cpu[i].uvw[2][r] = Addition(9);
                    cpu[i].uvw[2][r].set(q,-1);
                    cpu[i].uvw[2][0].set(q,1);
                    cpu[i].copyTo(cpu[i]);
                }
                cpu[i].copyTo(gpu[i]);
                expected[i] = states[i] = {uint32_t(100 + i)};
                switch (operation) {
                    case 0: for (int j = 0; j < 32; j++) cpu[i].tryFlip(expected[i]); break;
                    case 1: require(cpu[i].tryPlus(expected[i]), "plus fixture must take a plus edge"); break;
                    case 2: cpu[i].tryExpand(2, expected[i]); break;
                    case 3: require(cpu[i].tryReduce(), "reduction fixture must reduce rank"); break;
                    case 4: cpu[i].tryProject(expected[i]); break;
                    case 5: cpu[i].tryExtend(expected[i]); break;
                    case 6: cpu[i].tryProduct(expected[i]); break;
                    case 7: cpu[i].swapBasis(expected[i]); break;
                    case 8: cpu[i].swapSize(expected[i]); break;
                    case 9: cpu[i].tryMerge(*other, expected[i]); break;
                    case 10: cpu[i].tryProduct(*other); break;
                }
                require(cpu[i].validate(), "CPU transformation invalid: " + std::to_string(operation));
            }
            metalDispatch("transformationKernel", 4, 1, gpu, states, other, operation);
            for (int i = 0; i < 4; i++) {
                require(equal(cpu[i], gpu[i]), "CPU/Metal transformation mismatch: " + std::to_string(operation));
                require(states[i].value == expected[i].value, "CPU/Metal random decisions mismatch");
                gpu[i].save((output / ("transform-" + std::to_string(operation) + "-" + std::to_string(i) + ".json")).string());
            }
        }
        metalFree(gpu); metalFree(other); metalFree(states);
        U *reducers;
        metalAllocate(&reducers, 7 * sizeof(U));
        metalAllocate(&states, 7 * sizeof(RandomState));
        std::vector<U> cpu(7);
        RandomState expected[7];
        int values[][9] = {{1,1,-1,0,1,0,0,0,0}, {1,1,-1,1,0,0,0,0,0}, {-1,-1,1,0,0,1,0,0,0}, {0,1,-1,0,1,0,1,0,0}};
        for (int mode = 0; mode < 7; mode++) {
            reducers[mode].clear();
            for (auto &row : values) require(reducers[mode].addExpression(row,9), "reducer input rejected");
            reducers[mode].setMode(mode);
            cpu[mode].copyFrom(reducers[mode]);
            expected[mode] = states[mode] = {uint32_t(331 + mode)};
            cpu[mode].reduce(expected[mode]);
        }
        metalDispatch("reducerTestKernel", 7, 1, reducers, states);
        for (int mode = 0; mode < 7; mode++) {
            std::ostringstream a,b;
            cpu[mode].write(a,"u","    "); reducers[mode].write(b,"u","    ");
            require(a.str() == b.str(), "CPU/Metal reducer circuit mismatch: " + std::to_string(mode));
            require(cpu[mode].getAdditions() == reducers[mode].getAdditions(), "reducer count mismatch");
            require(states[mode].value == expected[mode].value, "reducer RNG mismatch");
        }
        int *capacity;
        metalAllocate(&capacity, sizeof(int));
        metalDispatch("capacityTestKernel", 1, 1, reducers, capacity);
        require(capacity[0] == 1, "GPU reducer must reject an oversized expression");
        metalFree(capacity);
        int zero[9] = {};
        cpu[0].clear();
        require(cpu[0].addExpression(zero,9) && cpu[0].getAdditions() == 0 && cpu[0].getNaiveAdditions() == 0, "zero expression count must be zero");
        metalFree(reducers); metalFree(states);
        std::cout << "PASS: 576 arithmetic cases, 64 random additions, 44 transformations, seven reducer modes, capacity, RNG and layouts\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
