import os
import numpy as np

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Capacity values for capa, capb, capc files from Table 1 in the paper
CAPA_CAPACITIES = [8000, 10000, 12000, 14000]
CAPB_CAPACITIES = [5000, 6000, 7000, 8000]
CAPC_CAPACITIES = [4000, 5000, 6000, 7000]

def read_cap_matrix(file_path):
    """
    Read a CAP data file and return the problem as matrices.
    
    Args:
        file_path (str): Path to the CAP data file
        
    Returns:
        tuple: (m, n, fixed_costs, transport_costs)
            - m (int): Number of potential warehouse locations
            - n (int): Number of customers
            - fixed_costs (np.ndarray): Array of fixed costs for each warehouse
            - transport_costs (np.ndarray): Matrix of transportation costs (n x m)
    """

    if "capa" in file_path:
        return read_special_cap_file(file_path, CAPA_CAPACITIES[0])
    elif "capb" in file_path:
        return read_special_cap_file(file_path, CAPB_CAPACITIES[0])
    elif "capc" in file_path:
        return read_special_cap_file(file_path, CAPC_CAPACITIES[0])

    with open(file_path, 'r') as f:
        # Read first line to get problem size
        m, n = map(int, f.readline().strip().split())
        
        # Read fixed costs (skip capacity values)
        fixed_costs = np.zeros(m)
        for i in range(m):
            _, fixed_costs[i] = map(float, f.readline().strip().split())
            
        # Read transportation costs for each customer
        transport_costs = np.zeros((n, m))
        for j in range(n):
            # Skip demand value
            _ = float(f.readline().strip())
            # Read transportation costs, handling potential line breaks
            line = f.readline().strip()
            while len(line.split()) < m:  # Keep reading until we have m values
                line += " " + f.readline().strip()
            transport_costs[j] = list(map(float, line.split()))
            
    return m, n, fixed_costs, transport_costs

def read_special_cap_file(file_path, capacity):
    """
    Read a special CAP file (capa, capb, or capc) with a specific capacity value.
    
    Args:
        file_path (str): Path to the special CAP file
        capacity (float): Capacity value to use (replaces 'capacity' in the file)
        
    Returns:
        tuple: (m, n, fixed_costs, transport_costs)
            - m (int): Number of potential warehouse locations
            - n (int): Number of customers
            - fixed_costs (np.ndarray): Array of fixed costs for each warehouse
            - transport_costs (np.ndarray): Matrix of transportation costs (n x m)
    """
    with open(file_path, 'r') as f:
        # Read first line to get problem size
        m, n = map(int, f.readline().strip().split())
        
        # Read fixed costs (replace 'capacity' with actual value)
        fixed_costs = np.zeros(m)
        for i in range(m):
            line = f.readline().strip()
            if 'capacity' in line.lower():
                line = line.replace('capacity', str(capacity))
            _, fixed_costs[i] = map(float, line.split())
            
        # Read transportation costs for each customer
        transport_costs = np.zeros((n, m))
        for j in range(n):
            # Skip demand value
            _ = float(f.readline().strip())
            # Read transportation costs, handling potential line breaks
            line = f.readline().strip()
            while len(line.split()) < m:  # Keep reading until we have m values
                line += " " + f.readline().strip()
            transport_costs[j] = list(map(float, line.split()))
            
    return m, n, fixed_costs, transport_costs

def print_matrix_example(m, n, fixed_costs, transport_costs):
    """
    Print a formatted example of the CAP instance matrices.
    
    Args:
        m (int): Number of potential warehouse locations
        n (int): Number of customers
        fixed_costs (np.ndarray): Array of fixed costs
        transport_costs (np.ndarray): Matrix of transportation costs
    """
    print(f"\nCAP Instance: {m} warehouses × {n} customers")
    print("\nFixed Costs:")
    print("-" * 50)
    for i, cost in enumerate(fixed_costs):
        print(f"Warehouse {i+1:2d}: {cost:10.2f}")
    
    print("\nTransportation Costs Matrix:")
    print("-" * 50)
    # Print header
    print("Customer", end="")
    for i in range(m):
        print(f" | W{i+1:2d}", end="")
    print("\n" + "-" * (8 + m * 7))
    
    # Print matrix
    for j in range(n):
        print(f"{j+1:7d}", end="")
        for i in range(m):
            print(f" | {transport_costs[j,i]:5.2f}", end="")
        print()

def main():
    """Example usage of the CAP matrix reader."""
    # Example with a small instance
    print("Example CAP Instance (7 warehouses × 5 customers):")
    print("=" * 50)
    
    # Create a small example instance
    m, n = 7, 5
    fixed_costs = np.array([1000, 1200, 800, 1500, 900, 1100, 1300])
    transport_costs = np.array([
        [50, 60, 40, 70, 45, 55, 65],
        [45, 55, 35, 65, 40, 50, 60],
        [60, 70, 50, 80, 55, 65, 75],
        [40, 50, 30, 60, 35, 45, 55],
        [55, 65, 45, 75, 50, 60, 70]
    ])
    
    print_matrix_example(m, n, fixed_costs, transport_costs)
    
    # Example with a real instance
    print("\nReal CAP Instance (cap71):")
    print("=" * 50)
    
    try:
        f = os.path.join(ROOT_DIR, "data/cap/cap71.txt")
        m, n, fixed_costs, transport_costs = read_cap_matrix(f)
        print_matrix_example(m, n, fixed_costs, transport_costs)
    except Exception as e:
        print(f"Error reading cap71.txt: {str(e)}")
        
    # Example with a special instance (capa)
    print("\nSpecial CAP Instance (capa with capacity 8000):")
    print("=" * 50)
    
    try:
        f = os.path.join(ROOT_DIR, "data/cap/capa.txt")
        m, n, fixed_costs, transport_costs = read_special_cap_file(f, CAPA_CAPACITIES[0])
        print_matrix_example(m, n, fixed_costs, transport_costs)
    except Exception as e:
        print(f"Error reading capa: {str(e)}")

if __name__ == "__main__":
    main() 