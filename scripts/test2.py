import time

control_freq = 200


def foo():
    time.sleep(1 / 1000)


def bar():
    pass
    # time.sleep(1 / 500)


times = []
try:
    while True:
        start = time.time()
        times.append(start)

        foo()
        bar()

        took = time.time() - start
        # print(took)
        time.sleep(max(0, 1 / control_freq - took))
except KeyboardInterrupt:
    if times:
        elapsed = time.time() - times[0]
        mean_frequency = (len(times) - 1) / elapsed if elapsed > 0 else 0
        print(f"Mean frequency: {mean_frequency:.2f} Hz")
    else:
        print("No data to calculate mean frequency.")
