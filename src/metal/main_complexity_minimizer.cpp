

int runProgram(int argc, char* argv[]) {
    ArgParser parser("complexity_minimizer", "Find best complexity of the fast matrix multiplication scheme in parallel using Metal");

    parser.add("--schemes", ArgType::Natural, "INT", "number of schemes", "1024");
    parser.add("--max-iterations", ArgType::Natural, "INT", "number of flips per iterations", "10000");
    parser.add("--path", ArgType::String, "PATH", "path to save schemes", "schemes");
    parser.add("--input-path", ArgType::String, "PATH", "path to init scheme(s)", "");
    parser.add("--block-size", ArgType::Natural, "INT", "number of Metal threads", "32");
    parser.add("--target-complexity", ArgType::Natural, "INT", "target complexity", "0");
    parser.add("--max-no-improvements", ArgType::Natural, "INT", "max iterations without improvements", "3");
    parser.add("--seed", ArgType::Natural, "INT", "random seed", "0");

    parser.add("--rounds", ArgType::Natural, "INT", "stop after this many outer iterations (0 is unbounded)", "0");
    if (!parser.parse(argc, argv))
        return argc == 2 && std::string(argv[1]) == "--help" ? 0 : 2;
    metalRounds = std::stoi(parser.get("--rounds"));
    validateOptions(parser);

    int schemesCount = std::stoi(parser.get("--schemes"));
    int maxIterations = std::stoi(parser.get("--max-iterations"));
    std::string path = parser.get("--path");
    std::string inputPath = parser.get("--input-path");
    int blockSize = std::stoi(parser.get("--block-size"));
    int targetComplexity = std::stoi(parser.get("--target-complexity"));
    int maxNoImprovements = std::stoi(parser.get("--max-no-improvements"));
    int seed = std::stoi(parser.get("--seed"));

    if (seed == 0)
        seed = time(0);

    std::ifstream f(inputPath);
    if (!f) {
        std::cout << "Unable to open file \"" << inputPath << "\"" << std::endl;
        return -1;
    }

    std::cout << "Start complexity minimization algorithm" << std::endl;
    std::cout << "- schemes count: " << schemesCount << std::endl;
    std::cout << "- max iterations: " << maxIterations << std::endl;
    std::cout << "- path: " << path << std::endl;
    std::cout << "- block size: " << blockSize << std::endl;
    std::cout << "- target complexity: " << targetComplexity << std::endl;
    std::cout << "- max no improvements: " << maxNoImprovements << std::endl;
    std::cout << "- seed: " << seed << std::endl;

    std::filesystem::create_directories(path);
    ComplexityMinimizer minimizer(schemesCount, blockSize, maxIterations, path, seed);

    bool result = minimizer.read(f);
    f.close();

    if (!result) {
        printf("Read invalid scheme\n");
        return -1;
    }

    try {
        minimizer.minimize(targetComplexity, maxNoImprovements);
        std::cout << "Success!" << std::endl;
    }
    catch (std::runtime_error e) {
        throw;
    }

    return 0;
}
