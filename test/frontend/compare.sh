python ./matmul_original.py > matmul_ir_original.pto
python ./matmul_pythonic.py > matmul_ir.pto

diff matmul_ir.pto matmul_ir_original.pto
# output should be empty if two IR are identical

python ./matmul_pythonic_ast.py > matmul_ir_ast.pto
diff matmul_ir.pto matmul_ir_ast.pto
