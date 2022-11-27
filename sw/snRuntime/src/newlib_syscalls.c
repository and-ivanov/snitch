#include <errno.h>
#undef errno
extern int errno;

extern void snrt_putchar(char character);

//#include "printf.h"
// void myprnt1(const char* ptr, int len) {
//   for (int i = 0; i < len; i++) {
//     snrt_putchar(ptr[i]);
//   }
// }

//#define myprnt(x) myprnt1(x, sizeof(x)-1)
#define myprnt(...)
//#define myprnt printf

void
_exit (int rc)
{
  myprnt("_exit\n");
  /* Convince GCC that this function never returns.  */
  for (;;)
    ;
}

//Close a file. Minimal implementation:

int _close(int file) {
  myprnt("_close\n");
  return -1;
}

//A pointer to a list of environment variables and their values. For a minimal environment, this empty list is adequate:

char *__env[1] = { 0 };
char **environ = __env;

// Transfer control to a new process. Minimal implementation (for a system without processes):

#include <errno.h>
#undef errno
extern int errno;
int _execve(char *name, char **argv, char **env) {
  myprnt("_execve\n");
  errno = ENOMEM;
  return -1;
}

// Create a new process. Minimal implementation (for a system without processes):

#include <errno.h>
#undef errno
extern int errno;
int _fork(void) {
  myprnt("_fork\n");
  errno = EAGAIN;
  return -1;
}

// Status of an open file. For consistency with other minimal implementations in these examples, all files are regarded as character special devices. The sys/stat.h header file required is distributed in the include subdirectory for this C library.

#include <sys/stat.h>
int _fstat(int fildes, struct stat *st) {
  myprnt("_fstat\n");
  // return -1;
  errno = ENOSYS;
  return -1;
  //st->st_mode = S_IFCHR;

  return 0;
}

// Process-ID; this is sometimes used to generate strings unlikely to conflict with other processes. Minimal implementation, for a system without processes:

int _getpid(void) {
  myprnt("_getpid\n");
  return 1;
}

// Query whether output stream is a terminal. For consistency with the other minimal implementations, which only support output to stdout, this minimal implementation is suggested:

int _isatty(int file) {
  myprnt("_isatty\n");
  return 1;
}

// Send a signal. Minimal implementation:

#include <errno.h>
#undef errno
extern int errno;
int _kill(int pid, int sig) {
  myprnt("_kill\n");
  errno = EINVAL;
  return -1;
}

// Establish a new name for an existing file. Minimal implementation:

#include <errno.h>
#undef errno
extern int errno;
int _link(char *old, char *new) {
  myprnt("_link\n");
  errno = EMLINK;
  return -1;
}

// Set position in a file. Minimal implementation:

int _lseek(int file, int ptr, int dir) {
  myprnt("_lseek\n");
  return 0;
}

// Open a file. Minimal implementation:

int _open(const char *name, int flags, int mode) {
  myprnt("_open\n");
  return -1;
}

// Read from a file. Minimal implementation:

int _read(int file, char *ptr, int len) {
  myprnt("_read\n");
  return 0;
}

// Increase program data space. As malloc and related functions depend on this, it is useful to have a working implementation. The following suffices for a standalone system; it exploits the symbol _end automatically defined by the GNU linker.

void *
_sbrk(int incr)
{
  myprnt("_sbrk\n");

  extern char   _end; /* Set by linker.  */
  static char * heap_end;
  char *        prev_heap_end;

  if (heap_end == 0) {
    heap_end = & _end;
  }

  prev_heap_end = heap_end;
  heap_end += incr;

  //printf("_sbrk: inc from %p to %p (incr=%d)\n", prev_heap_end, heap_end, incr);

  return (void *) prev_heap_end;
}

// Status of a file (by name). Minimal implementation:

int
_stat(const char  *file,
        struct stat *st)
{
  myprnt("_stat\n");
  st->st_mode = S_IFCHR;
  return 0;
}

// Timing information for current process. Minimal implementation:

int _times(struct tms *buf) {
  myprnt("_times\n");
  return -1;
}

// Remove a file’s directory entry. Minimal implementation:

#include <errno.h>
#undef errno
extern int errno;
int _unlink(char *name) {
  myprnt("_unlink\n");
  errno = ENOENT;
  return -1; 
}

// Wait for a child process. Minimal implementation:

#include <errno.h>
#undef errno
extern int errno;
int _wait(int *status) {
  myprnt("_wait\n");
  errno = ECHILD;
  return -1;
}

// Write to a file. libc subroutines will use this system routine for output to all files, including stdout—so if you need to generate any output, for example to a serial port for debugging, you should make your minimal write capable of doing this. The following minimal implementation is an incomplete example; it relies on a outbyte subroutine (not shown; typically, you must write this in assembler from examples provided by your hardware manufacturer) to actually perform the output.

int _write(int file, char *ptr, int len) {
  myprnt("_write\n");

  for (int i = 0; i < len; i++) {
    snrt_putchar(ptr[i]);
  }

  return len;
}


// #define ATOMIC_CAS(dst, exp, src) do { \
//   typeof(exp) e = 0; \
//   while (!__atomic_compare_exchange_n( \
//     &dst, &e, src, \
//     0, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE \
//   )) { e = exp; } \
// } while (0) 

// #define ATOMIC_SET(dst, src) do { \
//   __atomic_store_n(&dst, src, __ATOMIC_RELEASE); \
// } while (0)



// static int alloc_thread_id_plus1;
// static int num_alloc_locks;

// #include <stdint.h>
// uint32_t snrt_cluster_core_idx();

// void
// __malloc_lock ( struct _reent *_r ) {
//   myprnt("__malloc_lock\n");
//   uint32_t tid = snrt_cluster_core_idx();
//   if (alloc_thread_id_plus1 != tid + 1) {
//     ATOMIC_CAS(alloc_thread_id_plus1, 0, tid + 1);
//   }
//   num_alloc_locks++;
// }

// void
// __malloc_unlock ( struct _reent *_r ) {
//   myprnt("__malloc_unlock\n");
//   num_alloc_locks--;
//   if (num_alloc_locks == 0) {
//     ATOMIC_SET(alloc_thread_id_plus1, 0);
//   }
// }
