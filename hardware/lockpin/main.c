//
// Created by terry on 2025-05-08. FOR TESTING OF lockpin.c ONLY
//
#include "main.h"
#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
//int main(){
//    baud = 9600;
//    const char* serialport = "/dev/ttyS3";
//    init_lockpin(serialport, baud);
    ///for testing:

//    unlock(2000);
//    sleep(2);
//    lock(2000);
//    sleep(2);
//    unlock(2000);
//    sleep(2);
//    lock(20000);
//    sleep(2);
//    unlock(20000);
//    sleep(2);
//    stop();
//    close_lockpin();

//}

///call the following code for the actual use case
///add feedback variables if needed

int start(char* serial, int baudrate);
void call_lock(int action, int duration);
void end();

void lock_telecope(char* serial, int baudrate){
    int action = 0;
    int duration = 0;
    if (start(serial, baudrate)){
        printf("Enter: action,duration\nTo Exit: 9,9\n");
        while (1){
            scanf("%d,%d", &action, &duration);
            if (action == 9 && duration == 9)
                break;
            call_lock(action, duration);
        }
        end();
    } else {
        printf("ERROR: CANNOT OPEN SERIAL PORT\n");
    }
}

int start(char* serial, int baudrate){
    baud = baudrate;

    init_lockpin();

    if (serialport != NULL) {
        free(serialport);
        serialport = NULL;
    }

    /// Allocate and copy the input string
    serialport = malloc(strlen(serial) + 1);
    if (serialport != NULL) {
        strcpy(serialport, serial);
        return 1;
    }
    return 0;
}

void call_lock(int action, int duration){

    if (action == 0){
        unlock(duration);
    } else if (action == 1){
        lock(duration);
    } else if (action == 2){
        stop();
    }
}

void end(){
    close_lockpin();
}