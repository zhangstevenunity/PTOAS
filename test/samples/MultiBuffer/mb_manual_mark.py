from textwrap import dedent


if __name__ == "__main__":
    print(
        dedent(
            """
            module {
              func.func @mb_manual_mark(%arg0: memref<32x32xf16, #pto.address_space<gm>>,
                                        %arg1: memref<32x32xf16, #pto.address_space<gm>>) {
                %ub = memref.alloc() : memref<32x32xf16, #pto.address_space<vec>>
                %ub_mb = pto.mark_multibuffer %ub {num = 2 : i32}
                  : memref<32x32xf16, #pto.address_space<vec>> -> memref<32x32xf16, #pto.address_space<vec>>
                %c0 = arith.constant 0 : index
                %c1 = arith.constant 1 : index
                %c2 = arith.constant 2 : index
                scf.for %iv = %c0 to %c2 step %c1 {
                  pto.load_dps ins(%arg0 : memref<32x32xf16, #pto.address_space<gm>>)
                               outs(%ub_mb : memref<32x32xf16, #pto.address_space<vec>>)
                  pto.store_dps ins(%ub_mb : memref<32x32xf16, #pto.address_space<vec>>)
                                outs(%arg1 : memref<32x32xf16, #pto.address_space<gm>>)
                  scf.yield
                }
                return
              }
            }
            """
        ).lstrip()
    )
