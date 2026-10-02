// Runs async tasks one at a time, in submission order, and counts the ones
// not yet finished so a producer can hold back (waitForAtMost).
//
// After a task rejects, the tasks queued behind it are skipped, and drain()
// rejects with that first failure.
class SerialQueue {
    private tail: Promise<void> = Promise.resolve();

    private firstError: Error | null = null;

    private waiters: (() => void)[] = [];

    // queued tasks not yet finished
    pending = 0;

    push(task: () => Promise<void>) {
        this.pending++;
        this.tail = this.tail
            .then(() => (this.firstError ? undefined : task()))
            .catch((err: Error) => {
                this.firstError ??= err;
            })
            .finally(() => {
                this.pending--;
                const waiters = this.waiters;
                this.waiters = [];
                waiters.forEach((wake) => wake());
            });
    }

    async waitForAtMost(count: number) {
        while (this.pending > count) {
            await new Promise<void>((resolve) => {
                this.waiters.push(resolve);
            });
        }
    }

    async drain() {
        await this.tail;
        if (this.firstError) {
            throw this.firstError;
        }
    }
}

export { SerialQueue };
