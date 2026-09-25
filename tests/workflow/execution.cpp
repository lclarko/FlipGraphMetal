#include "../../src/metal/host.h"
#include "execution_layout.h"

int main(int argc, char **argv) {
    try {
        return fgm::runConfigured(argc, argv, "", "", nativeExecutionLayout());
    } catch (const fgm::Resource &error) {
        std::cerr << "resource_limit: " << error.what() << '\n';
        return 2;
    } catch (const std::exception &error) {
        std::cerr << "error: " << error.what() << '\n';
        return 1;
    }
}
