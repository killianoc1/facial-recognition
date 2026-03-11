import warnings
from scanner import register_face, recognize_face
import database
from recognizer import FaceRecognizer

warnings.filterwarnings("ignore", category=UserWarning, module='google.protobuf')


def delete_face():
    users = database.get_all_users()
    if not users:
        print("No registered faces.")
        return

    print("\nRegistered:")
    for i, (name, _) in enumerate(users, 1):
        print(f"{i}. {name}")

    try:
        num = int(input("\nDelete number (0=cancel): "))
        if num == 0:
            return
        if 1 <= num <= len(users):
            name = users[num-1][0]
            conn = database.connect()
            cursor = conn.cursor()
            cursor.execute("DELETE FROM users WHERE name = ?", (name,))
            conn.commit()
            conn.close()
            print(f"Deleted: {name}")
        else:
            print("Invalid.")
    except ValueError:
        print("Enter a number.")


def main():
    database.create_table()
    recognizer = FaceRecognizer()

    while True:
        print("\n===== Facial Recognition System =====")
        print("1. Register Face")
        print("2. Recognize Face")
        print("3. Delete Face")
        print("4. Exit")

        choice = input("Choose: ").strip()

        if choice == "1":
            register_face(recognizer)
        elif choice == "2":
            recognize_face(recognizer)
        elif choice == "3":
            delete_face()
        elif choice == "4":
            print("Goodbye!")
            break
        else:
            print("Invalid choice.")


if __name__ == "__main__":
    main()