#pragma once
#include "scheme_io.h"
#include <functional>
#include <memory>

namespace fgm {
struct JournalLimits {
    uint64_t storageBytes=268435456, maxTransactionBytes=1048576, indexMemoryBytes=1048576;
};
struct CommitReceipt {
    uint64_t sequence=0, offset=0, bytes=0;
    std::string hash;
    std::vector<std::string> novelIds, creditedIds;
    // Read-only replay can inspect complete frames beyond the durable head.
    // Those frames become acknowledged only after successful writable recovery.
    bool acknowledged=true;
};
struct JournalRecovery {
    uint64_t sequence=0, committedBytes=0, incompleteTailBytes=0, admissionCount=0;
    std::string hash;
    std::vector<std::filesystem::path> retainedTails;
    bool derivedIndexCurrent=false;
};
// Journal framing provides durability and historical identity accounting, not
// tensor verification. The controller must verify observations before append.
// Transactions require admissions:[{scheme_id,origin:"import"|"discovery",...}].
// Extra controller fields (including rank/domain/provenance) survive unchanged.
// The reserved _journal field is produced here and replayed to the controller.
// On disk: FGMJNL1 newline, then frames with an 8-byte version tag, 16-hex
// sequence and payload length, previous SHA256 and header SHA256. The bounded
// JSON payload is followed by FGMEND01, payload SHA256 and frame SHA256. Full
// malformed frames fail closed; only a structurally incomplete last frame is
// preserved as tail evidence and removed by writable recovery. Required
// committed-head.json binds the minimum acknowledged sequence/hash/byte offset;
// it must match retained history before any repair. Commit acknowledgement follows
// durable atomic publication of that watermark. A consistent rollback of journal
// and watermark together requires an independently retained head to detect.
class Journal {
public:
    using Visitor=std::function<void(const Json&,const CommitReceipt&)>;
    struct Reservation {
    private:
        uint64_t payloadBytes=0, storageBytes=0, sequence=0;
        const Journal *owner=nullptr;
        friend class Journal;
    };
    Journal(const std::filesystem::path&,JournalLimits,bool create=false);
    ~Journal();
    Journal(const Journal&)=delete;
    Journal& operator=(const Journal&)=delete;
    // Returned state views remain valid until the next journal mutation.
    const JournalRecovery &recover(const Visitor& visitor={});
    // No file creation, repair, index rebuilding or writable opens. Historical
    // credit is checked by bounded read-only passes over authoritative frames.
    static JournalRecovery replayReadOnly(const std::filesystem::path&,JournalLimits,
                                          const Visitor& visitor={},
                                          const std::function<void(uint64_t)>& accountRead={});
    // payloadBytes bounds the final stored JSON, including generated accounting.
    // Reservation includes worst-case bounded index-recovery scratch and a tail
    // evidence copy; it is invalidated by any successful intervening append.
    Reservation reserve(uint64_t payloadBytes,uint64_t additionalScratchBytes=0);
    CommitReceipt append(const Json&,const Reservation&);
    bool contains(const std::string& schemeId) const;
    void rebuildIndex();
    const JournalRecovery &state() const;
#ifdef FGM_JOURNAL_TESTING
    // Test builds only. A positive write cap exercises short-write completion;
    // failAfterBytes and failSyncAfter count down before injecting EIO.
    static void testFaults(int64_t failAfterBytes=-1,int64_t failSyncAfter=-1,size_t writeCap=0);
    struct TestIoEvent { std::string operation; std::filesystem::path path; bool succeeded; };
    // Paths are resolved from the actual synchronized descriptor. Publication
    // events follow the rename, so tests can inspect file-specific ordering.
    static void testIoReset(const std::filesystem::path &failSyncPath={});
    static std::vector<TestIoEvent> testIoEvents();
#endif
private:
    struct Impl;
    std::unique_ptr<Impl> impl;
};
}
