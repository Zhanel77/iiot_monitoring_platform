from app.consumer import MQTTConsumer


def main() -> None:
    consumer = MQTTConsumer()
    consumer.start()


if __name__ == "__main__":
    main()