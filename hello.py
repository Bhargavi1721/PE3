# GitHub Basic Operations - Sample Program

def greet_user(name):
    print(f"Hello, {name}!")
    print("Welcome to GitHub Basic Operations.")

def calculate_sum(a, b):
    return a + b

def check_number(number):
    if number > 0:
        return "Positive"
    elif number < 0:
        return "Negative"
    else:
        return "Zero"

# Program execution
name = "Hyma"
num1 = 25
num2 = 15

greet_user(name)

total = calculate_sum(num1, num2)
print(f"\nFirst number: {num1}")
print(f"Second number: {num2}")
print(f"Sum: {total}")

result = check_number(total)
print(f"The sum is {result}.")

print("\nGitHub operations completed successfully!")
print("Repository: PE3")
print("Branch: development")
print("Code uploaded successfully.")
