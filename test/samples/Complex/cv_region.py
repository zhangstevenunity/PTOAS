from mlir.ir import Context, Location, Module, InsertionPoint, IndexType, F32Type
from mlir.dialects import func, arith, pto

def build_sections_test():
    with Context() as ctx:
        # Register the PTO Dialect
        pto.register_dialect(ctx, load=True)

        with Location.unknown(ctx):
            m = Module.create()

            # --- 1. Prepare types ---
            f32 = F32Type.get(ctx)
            ptr_f32 = pto.PtrType.get(f32, ctx)  # !pto.ptr<f32>
            
            # Define Tile View and Buffer types
            # Assume a 32x32 float tile
            tv2_f32 = pto.TensorViewType.get(2, f32, ctx)  # !pto.tensor_view<2xf32>
            tile_view_32 = pto.PartitionTensorViewType.get([32, 32], f32, ctx)
            
            # Tile Buffer configuration
            ub = pto.AddressSpaceAttr.get(pto.AddressSpace.VEC, ctx)
            bl = pto.BLayoutAttr.get(pto.BLayout.RowMajor, ctx)
            sl = pto.SLayoutAttr.get(pto.SLayout.NoneBox, ctx)
            pd = pto.PadValueAttr.get(pto.PadValue.Null, ctx)
            cfg = pto.TileBufConfigAttr.get(bl, sl, 512, pd, ctx)
            
            tile_buf_32 = pto.TileBufType.get([32, 32], f32, ub, [32, 32], cfg, ctx)

            # Function signature: (ptr, ptr) -> void
            fn_ty = func.FunctionType.get([ptr_f32, ptr_f32], [])

            with InsertionPoint(m.body):
                fn = func.FuncOp("test_cube_vector_sections", fn_ty)
                entry = fn.add_entry_block()

            with InsertionPoint(entry):
                # --- 2. Common code (outside any section) ---
                # Corresponds to the code outside #ifdef in C++
                
                # Constant definitions
                c0 = arith.ConstantOp(IndexType.get(ctx), 0).result
                c1 = arith.ConstantOp(IndexType.get(ctx), 1).result
                c32 = arith.ConstantOp(IndexType.get(ctx), 32).result

                arg0, arg1 = entry.arguments
                
                # Create Tensor Views (Global Memory Pointers)
                tv0 = pto.MakeTensorViewOp(tv2_f32, arg0, [c32, c32], [c32, c1]).result
                tv1 = pto.MakeTensorViewOp(tv2_f32, arg1, [c32, c32], [c32, c1]).result

                # Create Subviews (slicing)
                # Note: We keep the name PartitionViewOp as provided.
                sv0 = pto.PartitionViewOp(tile_view_32, tv0, offsets=[c0, c0], sizes=[c32, c32]).result
                sv1 = pto.PartitionViewOp(tile_view_32, tv1, offsets=[c0, c0], sizes=[c32, c32]).result

                # Allocate Tile Buffers (L1/UB)
                # These variables need to be accessible in both sections, so they are defined outside.
                tb0 = pto.AllocTileOp(tile_buf_32).result
                tb1 = pto.AllocTileOp(tile_buf_32).result

                # --- 3. CUBE Section ---
                # Corresponds to #if defined(__DAV_CUBE__)
                # Create Section Op
                cube_section = pto.SectionCubeOp()
                # SectionOp is initially empty; we manually add a block.
                cube_block = cube_section.body.blocks.append()
                
                with InsertionPoint(cube_block):
                    # Perform load (TLoad) on the Cube core
                    # pto.TLoadOp(sv0, tb0)
                    # Note: The parameter order of TLoadOp depends on your definition (src, dst) or (dst, src).
                    # Your C++ definition appears to be (src, dst, ...).
                    pto.TLoadOp(None, sv0, tb0)
                    
                    # Note: SectionOp has NoTerminator, so no yield or return is needed.

                # --- 4. VECTOR Section ---
                # Corresponds to #if defined(__DAV_VEC__)
                vec_section = pto.SectionVectorOp()
                vec_block = vec_section.body.blocks.append()
                
                with InsertionPoint(vec_block):
                    # Perform computation (Abs) and store (TStore) on the Vector core
                    
                    # TAbs: dst = abs(src) -> tb1 = abs(tb0)
                    pto.TAbsOp(tb0, tb1)
                    
                    # TStore: dst(GM) = src(UB) -> sv1 = tb1
                    # Again, note the parameter order; Store is typically (src, dst).
                    pto.TStoreOp(None, tb1, sv1)

                # --- 5. Function return ---
                func.ReturnOp([])

            # Verify that the generated IR is valid
            m.operation.verify()
            return m

if __name__ == "__main__":
    print(build_sections_test())