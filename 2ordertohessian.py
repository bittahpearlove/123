import sympy as sp
def compute_hessian(a=None, b=None, c=None, d=None, e=None, f=None, custom_function=None):
    x, y = sp.symbols('x y')
    if custom_function:
        function = custom_function
    else:
        function = a*x**2 + b*x*y + c*y**2 + d*x + e*y + f
    partial_x = sp.diff(function, x)
    partial_y = sp.diff(function, y)
    hessian_matrix = sp.Matrix([[sp.diff(partial_x, x), sp.diff(partial_x, y)],
                                 [sp.diff(partial_y, x), sp.diff(partial_y, y)]])
    print("Hessian Matrix:")
    sp.pprint(hessian_matrix)
    return hessian_matrix
def main():
    print("Enter coefficients for the quadratic function (ax^2 + bxy + cy^2 + dx + ey + f):")
    a = float(input("Enter coefficient a (for x^2): "))
    b = float(input("Enter coefficient b (for xy): "))
    c = float(input("Enter coefficient c (for y^2): "))
    d = float(input("Enter coefficient d (for x): "))
    e = float(input("Enter coefficient e (for y): "))
    f = float(input("Enter constant term f: "))
    custom_input = input("Would you like to enter a custom function instead? (yes/no): ").strip().lower()
    if custom_input == 'yes':
        custom_function_str = input("Enter your custom function in terms of x and y (e.g., 'x**2 + y**2 - 3*x*y'): ")
        custom_function = sp.sympify(custom_function_str)
        hessian_matrix = compute_hessian(custom_function=custom_function)
    else:
        hessian_matrix = compute_hessian(a, b, c, d, e, f)
    evaluate_at_point = input("Would you like to evaluate the Hessian at a specific point? (yes/no): ").strip().lower()
    if evaluate_at_point == 'yes':
        x0 = float(input("Enter x-coordinate: "))
        y0 = float(input("Enter y-coordinate: "))
        hessian_at_point = hessian_matrix.subs({sp.symbols('x'): x0, sp.symbols('y'): y0})
        print("\nHessian Matrix at ({}, {}):".format(x0, y0))
        sp.pprint(hessian_at_point)
if __name__ == "__main__":
    main()