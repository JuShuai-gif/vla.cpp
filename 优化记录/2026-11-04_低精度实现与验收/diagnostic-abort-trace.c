#define _GNU_SOURCE
#include <execinfo.h>
#include <signal.h>
#include <unistd.h>
static void trace_abort(int sig) {
    void *frames[64];
    const char header[]="DIAGNOSTIC_NATIVE_ABORT_BACKTRACE\n";
    write(2,header,sizeof(header)-1);
    int n=backtrace(frames,64);
    backtrace_symbols_fd(frames,n,2);
    _exit(128+sig);
}
__attribute__((constructor)) static void install_trace(void) {
    void *frames[2];backtrace(frames,2);
    struct sigaction action={0};
    action.sa_handler=trace_abort;action.sa_flags=SA_RESETHAND;
    sigemptyset(&action.sa_mask);
    sigaction(SIGABRT,&action,0);
}
