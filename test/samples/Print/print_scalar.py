"""Test pto.print: format is string attribute, scalar is operand. Generates PRINTF("format", scalar) in C++."""
from mlir.ir import Context, Location, Module, InsertionPoint
from mlir.dialects import func, pto
from mlir.ir import F32Type


def build():
    with Context() as ctx:
        pto.register_dialect(ctx, load=True)

        with Location.unknown(ctx):
            m = Module.create()

            f32 = F32Type.get(ctx)
            fn_ty = func.FunctionType.get([f32], [])
            with InsertionPoint(m.body):
                fn = func.FuncOp("print_scalar_kernel", fn_ty)
                entry = fn.add_entry_block()

            with InsertionPoint(entry):
                scalar = entry.arguments[0]
                pto.PrintOp("scalar = %+08.3f", scalar)
                func.ReturnOp([])

            m.operation.verify()
            return m


if __name__ == "__main__":
    print(build())
