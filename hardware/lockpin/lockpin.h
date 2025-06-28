void lock(int duration);
void unlock(int duration);
void init_lockpin();
void close_lockpin();
void stop();

extern int is_locked;
extern int fd;
extern int baud;
extern char* serialport;