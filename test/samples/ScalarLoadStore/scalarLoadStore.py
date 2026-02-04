from mlir.ir import Context, Location, Module, InsertionPoint
from mlir.dialects import func, arith, memref
from mlir.ir import F32Type, IndexType, MemRefType


def build():
    with Context() as ctx:
        with Location.unknown(ctx):
            m = Module.create()

            f32 = F32Type.get(ctx)
            idx = IndexType.get(ctx)
            m2_f32 = MemRefType.get([32, 32], f32)

            fn_ty = func.FunctionType.get([m2_f32, m2_f32], [])
            with InsertionPoint(m.body):
                fn = func.FuncOp("scalar_load_store_kernel_2d", fn_ty)
                entry = fn.add_entry_block()

            with InsertionPoint(entry):
                c0 = arith.ConstantOp(idx, 0).result
                arg0, arg1 = entry.arguments

                v = memref.LoadOp(arg0, [c0, c0]).result
                memref.StoreOp(v, arg1, [c0, c0])
                func.ReturnOp([])

            m.operation.verify()
            return m


if __name__ == "__main__":
    print(build())
