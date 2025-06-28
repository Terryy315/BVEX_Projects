#include "arduino-serial-lib.h"
#include <string.h>
#include <stdio.h>
int is_locked = 0;
int fd = -1;
char* serialport;
int baud;
void listen(int duration);
void lock(int duration){
    char message[50] = "1,";
    char time[20];
    sprintf(time, "%d", duration);
    strcat(message, time);
    strcat(message, "\0");
    serialport_write(fd, message);
    listen(duration);
    is_locked = 1;
}

void unlock(int duration){
    char message[50] = "0,";
    char time[20];
    sprintf(time, "%d", duration);
    strcat(message, time);
    strcat(message, "\0");
    serialport_write(fd, message);
    listen(duration);
    is_locked = 0;
}

//STOP MEANS STOP AND RESET
void stop(){
    char message[4] = "2,0";
    serialport_write(fd, message);
    is_locked = 0;
}

void init_lockpin(){
    fd = serialport_init(serialport, baud);
    if( fd != -1 )
        serialport_flush(fd);
}

void close_lockpin(){
    serialport_close(fd);
}

void listen(int duration){
    char buff[50];
    char eol = '\n';
    serialport_read_until(fd, buff, eol, 50, duration * 1.5);
    printf("%s\n", buff);
}