#pragma once
#include "json.h"
#include <array>
#include <filesystem>
#include <functional>
#include <memory>

namespace fgm {
struct Resource : std::runtime_error { using std::runtime_error::runtime_error; };
struct AdmissionLimits {
    uint64_t record=1048576, work=10000000, selection=67108864, scan=268435456;
};
using Matrix=std::vector<std::vector<int64_t>>;
struct SchemeRecord {
    std::array<uint32_t,3> n{};
    uint32_t rank=0;
    bool f2=false;
    std::array<Matrix,3> f;
    bool circuit=false;
    uint64_t operations=0;
    std::string sourceOrientation="cyclic-w";
    Json sourceMetadata=Json::dict();
};
struct AdmittedScheme {
    SchemeRecord source;
    SchemeRecord effective;
    Json report;
    uint64_t recordIndex=0;
};
// One context is one bounded operation. It is not shared across concurrent calls.
// Callbacks are provisional until the read method returns successfully: final
// source hash checks may reject the entire input after a callback has run.
class AdmissionContext {
public:
    using Callback=std::function<void(const AdmittedScheme&)>;
    explicit AdmissionContext(AdmissionLimits limits={});
    ~AdmissionContext();
    AdmissionContext(const AdmissionContext&)=delete;
    AdmissionContext& operator=(const AdmissionContext&)=delete;
    void readRecords(const std::filesystem::path&,const std::string& format,
                     const std::string& domain,const Callback&);
    void readSelection(const std::map<std::string,std::string>&,const Callback&);
    // fromJson/readRecords/readSelection reset work per parsed record; verify and
    // analyze charge the current record budget cumulatively.
    SchemeRecord fromJson(const Json&,const std::string& domain="");
    std::string identity(const SchemeRecord&,bool canonical);
    SchemeRecord normalized(const SchemeRecord&);
    bool verify(const SchemeRecord&);
    Json analyze(const SchemeRecord&);
    Json schemeJson(const SchemeRecord&);
    std::string sha256File(const std::filesystem::path&);
    void accountRead(uint64_t bytes);
    uint64_t scannedBytes() const;
    uint64_t workUsed() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl;
    friend int schemeToolMain(int,char**);
    friend std::string sha256Bytes(const std::string&);
    friend void publishNew(const std::filesystem::path&,const std::string&);
};
// Owned-storage accounting includes object sizes and container capacities. Map
// nodes include a conservative four-pointer allowance. Allocator bookkeeping,
// transient copies and process RSS are not bounded by these estimates.
uint64_t jsonMemoryBytes(const Json&);
uint64_t admittedMemoryBytes(const AdmittedScheme&);
std::string sha256Bytes(const std::string&);
void publishNew(const std::filesystem::path&,const std::string&);
int schemeToolMain(int argc,char**argv);
}
