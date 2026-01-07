from .verify import eda_tools
import re
import os
from multiprocessing import Process, Queue
import psutil
import hashlib
import random


def verify_one_sample(golden, dut_code, uid=None):
    uid = dut_code + str(random.randint(0,2147483647))
    uid = hashlib.md5(uid.encode("utf-8")).hexdigest()
    v = eda_tools(quiet=True)

    assert isinstance(golden, dict)
    gold_code = golden['code']
    port_info = (golden.get('input_port_width', None),
                    golden.get('output_port_width', None),
                    golden.get('clock_port_polarity', None),
                    golden.get('reset_port_polarity_sync', None))

    if not gold_code or not dut_code:
        return {"correct": False}

    try:
        gold_top = v.auto_top(gold_code)
        gate_top = v.auto_top(dut_code)
    except Exception as e:
        # exception in verification, gold code or dut code have syntax problems
        # print("Parse error:", e.args)
        return {"correct": False, "parse_error": e.args}

    gold_path, dut_path = f"./tmp/testcase/{uid}_gold.v", f"./tmp/testcase/{uid}_dut.v"
    test_path = f"./tmp/work/{uid}"
    
    try:
        if not os.path.exists("./tmp/testcase"):
            os.makedirs("./tmp/testcase", exist_ok=True)
        if not os.path.exists("./tmp/work"):
            os.makedirs("./tmp/work", exist_ok=True)
        if not os.path.exists(test_path):
            os.makedirs(test_path, exist_ok=True)
    finally:
        pass
    
    with open(gold_path, "w") as f:
        f.write(gold_code)
    with open(dut_path, "w") as f:
        f.write(dut_code)

    result = None
    try:
        equiv = v.equiv_with_testbench(
            gold_path,
            dut_path,
            gold_top,
            gate_top,
            test_path,
            port_info=port_info,
        )
    except Exception as e:
        # print("Test error:", e.args)
        result = {"correct": False, "test_error": e.args}
    finally:
        if os.path.exists(gold_path):
            os.remove(gold_path)
        if os.path.exists(dut_path):
            os.remove(dut_path)
        if os.path.exists(test_path):
            os.system(f"rm -r {test_path}")

    if result is None:
        result = {"correct": equiv[0], "error_rate": equiv[1], "detail": equiv[2]}
    return result


def kill_process_tree(pid):
    parent = psutil.Process(pid)
    children = parent.children(recursive=True)
    for child in children:
        child.terminate()
    parent.terminate()


def verify_one_sample_wrapper(args):
    def target(queue):
        result = verify_one_sample(*args)
        queue.put(result)

    queue = Queue()
    process = Process(target=target, args=(queue,))
    process.start()
    process.join(timeout=30)

    if process.is_alive():
        kill_process_tree(process.pid)
        process.join()
        print("Function timed out!")
        return {"correct": False, "timeout": True}
    else:
        return queue.get()


def _extract_module_blocks(code):
    """Extract only module...endmodule blocks from code, excluding testbenches.

    Returns:
        list: List of individual module strings (non-testbench modules)
        None: If all modules are testbenches or no modules found
        str: Original code if no module blocks could be parsed
    """
    # Remove comments first
    note_pattern = r"(//[^\n]*|/\*[\s\S]*?\*/)"
    code = re.sub(note_pattern, "", code)

    # Extract all module...endmodule blocks
    # Pattern matches: module <name> ... endmodule
    module_pattern = r"module\s+[a-zA-Z_][a-zA-Z0-9_$]*.*?endmodule"
    modules = re.findall(module_pattern, code, re.DOTALL)

    if modules:
        # Filter out testbench modules
        filtered_modules = []
        for module_block in modules:
            # Extract module name
            name_match = re.search(r"module\s+([a-zA-Z_][a-zA-Z0-9_$]*)", module_block)
            if name_match:
                module_name = name_match.group(1).lower()

                # Check if it's a testbench (case-insensitive)
                if not (
                    module_name.startswith("tb_") or
                    module_name.endswith("_tb") or
                    "testbench" in module_name or
                    "test_bench" in module_name
                ):
                    filtered_modules.append(module_block)
            else:
                # If we can't extract name, keep it (might be valid but complex syntax)
                filtered_modules.append(module_block)

        if filtered_modules:
            return filtered_modules  # Return list of individual modules
        else:
            # All modules were testbenches - return None to indicate no valid DUT code
            return None

    # If no module...endmodule blocks found at all, return original
    # (might have valid module but complex syntax)
    return code


def extract_verilog(verilog_code):
    """Extract Verilog module(s) from code with markdown formatting.

    Returns:
        list: List of individual module strings (non-testbench modules)
        None: If no valid modules found
    """
    # Try multiple markdown formats
    # Note: (?!\w) negative lookahead prevents "v" from matching "verilog"
    patterns = [
        r"```verilog\s*([\s\S]*?)\s*```",
        r"```v(?!\w)\s*([\s\S]*?)\s*```",         # Not followed by word char
        r"```systemverilog\s*([\s\S]*?)\s*```",
        r"```sv(?!\w)\s*([\s\S]*?)\s*```",        # Not followed by word char
    ]

    # Collect modules from ALL patterns, not just the first matching one
    all_modules = []

    for pattern in patterns:
        matches = re.findall(pattern, verilog_code)
        for match in matches:
            extracted = _extract_module_blocks(match)
            # Collect modules from this block
            if isinstance(extracted, list) and extracted:
                all_modules.extend(extracted)  # Add all modules from this block
            elif isinstance(extracted, str) and "module" in extracted:
                all_modules.append(extracted)  # Add single module

    # If we found any modules across all patterns and blocks, return them
    if all_modules:
        return all_modules

    # Fallback: Extract plain text with module keyword (Qwen3 format)
    if "module" in verilog_code and "endmodule" in verilog_code:
        # Remove <think> and <answer> tags if present
        cleaned = re.sub(r"<think>.*?</think>", "", verilog_code, flags=re.DOTALL)
        cleaned = re.sub(r"</?answer>", "", cleaned)
        extracted = _extract_module_blocks(cleaned.strip())
        if isinstance(extracted, list) and extracted:
            return extracted
        elif isinstance(extracted, str) and extracted:
            return [extracted]  # Wrap in list

    return None
