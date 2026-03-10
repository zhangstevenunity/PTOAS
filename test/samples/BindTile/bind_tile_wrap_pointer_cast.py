def main() -> None:
    # Late-stage IR: explicit UB addresses via pointer_cast. This is used to
    # regression-test that plain `pto.bind_tile` does *not* lower to `TRESHAPE`
    # (which is reserved for SSA `pto.treshape` view semantics).
    print(
        r"""
module {
  func.func @bind_tile_wrap_pointer_cast() {
    %c0_i64 = arith.constant 0 : i64
    %c4096_i64 = arith.constant 4096 : i64
    %c32 = arith.constant 32 : index
    %cst = arith.constant 3.14 : f32

    %a_mem = pto.pointer_cast(%c0_i64) : memref<32x32xf32, #pto.address_space<vec>>
    %b_mem = pto.pointer_cast(%c4096_i64) : memref<32x32xf32, #pto.address_space<vec>>

    %a_tile = pto.bind_tile %a_mem, %c32, %c32 {
      config = #pto.tile_buf_config<blayout=#pto.blayout<row_major>, slayout=#pto.slayout<none_box>, s_fractal_size=512, pad=#pto.pad_value<null>>
    } : memref<32x32xf32, #pto.address_space<vec>> -> memref<32x32xf32, #pto.address_space<vec>>

    %b_tile = pto.bind_tile %b_mem, %c32, %c32 {
      config = #pto.tile_buf_config<blayout=#pto.blayout<row_major>, slayout=#pto.slayout<none_box>, s_fractal_size=512, pad=#pto.pad_value<null>>
    } : memref<32x32xf32, #pto.address_space<vec>> -> memref<32x32xf32, #pto.address_space<vec>>

    pto.tadds ins(%a_tile, %cst : memref<32x32xf32, #pto.address_space<vec>>, f32)
             outs(%b_tile : memref<32x32xf32, #pto.address_space<vec>>)
    return
  }
}
"""
    )


if __name__ == "__main__":
    main()

