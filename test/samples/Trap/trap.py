"""Test pto.trap: generates TRAP() in C++."""
from mlir.ir import Context, Location, Module, InsertionPoint
from mlir.dialects import func, pto


def build():
    with Context() as ctx:
        pto.register_dialect(ctx, load=True)

        with Location.unknown(ctx):
            m = Module.create()

            fn_ty = func.FunctionType.get([], [])
            with InsertionPoint(m.body):
                fn = func.FuncOp("trap_kernel", fn_ty)
                entry = fn.add_entry_block()

            with InsertionPoint(entry):
                pto.TrapOp()
                func.ReturnOp([])

            m.operation.verify()
            return m


if __name__ == "__main__":
    print(build())
