// RUN: ptoas %s | FileCheck %s

module {
  func.func @sync_ops() {
    pto.record_event [#pto.pipe_event_type<TLOAD>, #pto.pipe_event_type<TVEC>, #pto.event<EVENT_ID0>]
    pto.wait_event [#pto.pipe_event_type<TLOAD>, #pto.pipe_event_type<TVEC>, #pto.event<EVENT_ID0>]
    return
  }
}

// CHECK: pto.set_flag
// CHECK: pto.wait_flag
