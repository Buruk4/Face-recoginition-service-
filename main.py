from core.registration import register_face
from core.recognition import check_face


def main():

    while True:

        print("\n==============================")
        print("     FACE RECOGNITION SYSTEM")
        print("==============================")

        print("1. Register New Person")
        print("2. Recognize Person")
        print("3. Exit")

        choice = input("\nChoose an option: ").strip()

        if choice == "1":

            register_face()

        elif choice == "2":

            check_face()

        elif choice == "3":

            print("Goodbye!")
            break

        else:

            print("Invalid option. Try again.")


if __name__ == "__main__":
    main()