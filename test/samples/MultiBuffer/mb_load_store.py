from textwrap import dedent


if __name__ == "__main__":
    print(
        dedent(
            """
            module {
              func.func @mb_load_store(%arg0: memref<32x32xf16, #pto.address_space<gm>>,
                                       %arg1: memref<32x32xf16, #pto.address_space<gm>>) {
                %c0 = arith.constant 0 : index
                %c1 = arith.constant 1 : index
                %c2 = arith.constant 2 : index
                scf.for %iv = %c0 to %c2 step %c1 {
                  %ub = memref.alloc() : memref<32x32xf16, #pto.address_space<vec>>
                  pto.load_dps ins(%arg0 : memref<32x32xf16, #pto.address_space<gm>>)
                               outs(%ub : memref<32x32xf16, #pto.address_space<vec>>)
                  pto.store_dps ins(%ub : memref<32x32xf16, #pto.address_space<vec>>)
                                outs(%arg1 : memref<32x32xf16, #pto.address_space<gm>>)
                  scf.yield
                }
                return
              }
            }
            """
        ).lstrip()
    )
