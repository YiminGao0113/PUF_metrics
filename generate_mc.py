# Parameters
output_csv = "output_mc_nand_expr_configurable.csv"
num_bits = 128          # Number of output bits
# iterations = 5          # Monte Carlo iterations (1-based)
# sample_time = "8e-8"    # Time point (e.g., 80 ns)
sample_time = "5e-8"    # Time point (e.g., 80 ns)

# Generate and write expressions
with open(output_csv, "w") as f:
    f.write("Name,Type,Output,EvalType,Plot,Save,Spec\n")
    
    for bit in range(num_bits):
        # for run in range(iterations):
        name = f"E{bit}"
        expr = f'value(getData("/out0<{bit}>" ?result "tran") {sample_time})'
        line = f"{name},expr,{expr},point,t,,\n"
        f.write(line)

print(f"✅ Wrote CSV with {num_bits} bits iterations at {sample_time}s.")
