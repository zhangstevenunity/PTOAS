from mlir.ir import Context, Location, Module, InsertionPoint
from mlir.dialects import func, arith, pto
from mlir.ir import F32Type, IndexType


def build():
    with Context() as ctx:
        pto.register_dialect(ctx, load=True)

        with Location.unknown(ctx):
            m = Module.create()

            f32 = F32Type.get(ctx)
            idx = IndexType.get(ctx)

            ptr_f32 = pto.PtrType.get(f32, ctx)

            fn_ty = func.FunctionType.get([ptr_f32, ptr_f32], [])
            with InsertionPoint(m.body):
                fn = func.FuncOp("ptr_scalar_rw", fn_ty)
                entry = fn.add_entry_block()

            with InsertionPoint(entry):
                c4 = arith.ConstantOp(idx, 4).result
                c8 = arith.ConstantOp(idx, 8).result

                src, dst = entry.arguments
                src_off = pto.addptr(src, c8)

                val = pto.load_scalar(f32, src_off, c4)
                pto.store_scalar(dst, c4, val)

                func.ReturnOp([])

            m.operation.verify()
            return m


if __name__ == "__main__":
    print(build())
