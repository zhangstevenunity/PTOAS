from mlir.ir import Context, Location, Module, InsertionPoint, F32Type, IndexType
from mlir.dialects import func, arith, pto


def build():
    with Context() as ctx:
        pto.register_dialect(ctx, load=True)

        with Location.unknown(ctx):
            m = Module.create()

            f32 = F32Type.get(ctx)
            idx = IndexType.get(ctx)
            ptr_f32 = pto.PtrType.get(f32, ctx)

            fn_ty = func.FunctionType.get([ptr_f32], [])
            with InsertionPoint(m.body):
                fn = func.FuncOp("test_scalar_intra_pipe_barrier", fn_ty)
                entry = fn.add_entry_block()

            with InsertionPoint(entry):
                ptr = entry.arguments[0]
                c0 = arith.ConstantOp(idx, 0).result
                c4 = arith.ConstantOp(idx, 4).result
                one = arith.ConstantOp(f32, 1.0).result
                two = arith.ConstantOp(f32, 2.0).result

                ptr_off = pto.addptr(ptr, c4)
                pto.store_scalar(ptr_off, c0, one)
                pto.store_scalar(ptr_off, c0, two)

                func.ReturnOp([])

            m.operation.verify()
            return m


if __name__ == "__main__":
    print(build())
