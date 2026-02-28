// RUN: ptoas --emit-manual-sync-as-event %s | FileCheck %s

module {
  func.func @sync_ops() {
    pto.record_event [#pto.pipe_event_type<TLOAD>, #pto.pipe_event_type<TVEC>, #pto.event<EVENT_ID0>]
    pto.wait_event [#pto.pipe_event_type<TLOAD>, #pto.pipe_event_type<TVEC>, #pto.event<EVENT_ID0>]
    return
  }
}

// CHECK: #include "pto/pto-inst.hpp"
// CHECK: template <typename EventT>
// CHECK: static inline void ptoas_record_event
// CHECK: __global__ AICORE void sync_ops()
// CHECK: Event<Op::TLOAD, Op::VECTOR>
// CHECK: ptoas_record_event
// CHECK: .Wait()
// CHECK-NOT: PTOAS__MANUAL_EVENT_WAIT
// CHECK-NOT: TSYNC
