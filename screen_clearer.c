#include <stdio.h>
#include <assert.h>
#include <malloc.h>
#include <sys/time.h>
#include <xcb/xcb.h>
#include <sys/shm.h>
#include <xcb/shm.h>
#include <immintrin.h>
#include <pthread.h>
#include <semaphore.h>
#include <errno.h>
#include <sys/sysinfo.h>

#define TRUE  1
#define FALSE 0

typedef uint8_t  u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;

typedef int8_t   s8;
typedef int16_t  s16;
typedef int32_t  s32;
typedef int64_t  s64;

typedef u8  b8;
typedef u16 b16;
typedef u32 b32;
typedef u64 b64;

typedef float  f32;
typedef double f64;

#define ENFORCE_READ_WRITE_ORDERING asm volatile("" ::: "memory")
#define ATOMIC_ADD(Value, Numeric) __sync_fetch_and_add(&Value, Numeric)
#define AVX2_TARGET __attribute__((target("avx2")))

volatile static int work_index;
volatile static int work_completed;
static sem_t work_semaphore;

static u32* pixels;
volatile static u32 pixel_count;
volatile static int alignment_magnitude;

void* AVX2_TARGET ThreadWork(void* args) {
   int thread_id = pthread_self();
   u32 pixel_color = 0xFF0000;

   for(;;) {
      sem_wait(&work_semaphore);
      int thread_work_index = ATOMIC_ADD(work_index, 1);

      u32 offset = pixel_count / alignment_magnitude;
      __m256i v8 = _mm256_set1_epi32(pixel_color);
      __m256i* pixel = (__m256i*)pixels + (offset * thread_work_index);

      for(u32 i = 0; i < offset; ++i) {
         _mm256_storeu_si256(pixel++, v8);
      }

      ATOMIC_ADD(work_completed, 1);
   }
}

int main() {
   int processor_count;

   alignment_magnitude = processor_count = get_nprocs_conf();
   alignment_magnitude *= 8;
   pthread_t threads[processor_count];

   sem_init(&work_semaphore, 0, 0);
   for(int i = 0; i < processor_count; ++i) {
      pthread_create(&threads[i], NULL, &ThreadWork, NULL);
   }

   xcb_connection_t* _connection = xcb_connect(0, 0);
   xcb_atom_t        _wm_protocols;
   xcb_atom_t        _wm_delete_protocol;

   xcb_screen_t* _screen = xcb_setup_roots_iterator(xcb_get_setup(_connection)).data;

   {
      xcb_intern_atom_cookie_t _intern_atom_cookie;
      xcb_intern_atom_reply_t* _intern_atom_reply;

      _intern_atom_cookie = xcb_intern_atom(_connection, 1, 12, "WM_PROTOCOLS");
      _intern_atom_reply  = xcb_intern_atom_reply(_connection, _intern_atom_cookie, 0);
      _wm_protocols       = _intern_atom_reply->atom;
      free(_intern_atom_reply);

      _intern_atom_cookie = xcb_intern_atom(_connection, 1, 16, "WM_DELETE_WINDOW");
      _intern_atom_reply  = xcb_intern_atom_reply(_connection, _intern_atom_cookie, 0);
      _wm_delete_protocol = _intern_atom_reply->atom;
      free(_intern_atom_reply);
   }

   xcb_window_t _window = xcb_generate_id(_connection);
   xcb_gcontext_t _gcontext = xcb_generate_id(_connection);
   {
      u32 mask     = XCB_CW_EVENT_MASK;
      u32 values[] = { XCB_EVENT_MASK_EXPOSURE | XCB_EVENT_MASK_STRUCTURE_NOTIFY };
      xcb_create_window(_connection,
                        XCB_COPY_FROM_PARENT,
                        _window,
                        _screen->root,
                        0, 0,
                        1280, 720,
                        0, XCB_WINDOW_CLASS_INPUT_OUTPUT, _screen->root_visual, mask, values);
      xcb_map_window(_connection, _window);
      xcb_change_property(_connection,
                          XCB_PROP_MODE_REPLACE,
                          _window,
                          _wm_protocols,
                          XCB_ATOM_ATOM,
                          32,
                          1, &_wm_delete_protocol);

      xcb_create_gc(_connection, _gcontext, _screen->root, 0, 0);

      xcb_flush(_connection);
   }

   xcb_pixmap_t _pixmap = 0;
   u32 _pixmap_mem = 0;
   xcb_shm_seg_t _pixmap_seg = 0;

   struct timeval tv_start_time;
   gettimeofday(&tv_start_time, 0);
   u16 width  = 0;
   u16 height = 0;
   while(TRUE) {
      b8 should_quit = FALSE;
      xcb_generic_event_t* e;
      while((e = xcb_poll_for_event(_connection))) {
         switch(e->response_type &~ 0x80) {
            #define CAST_DEF_EVENT(Type) xcb_##Type##_event_t* Type##_event = (xcb_##Type##_event_t*)e;
            case XCB_CLIENT_MESSAGE:
            {
               CAST_DEF_EVENT(client_message);
               if(client_message_event->data.data32[0] == _wm_delete_protocol) {
                  should_quit = TRUE;
               }
            } break;
            case XCB_CONFIGURE_NOTIFY:
            {
               CAST_DEF_EVENT(configure_notify);
               width  = configure_notify_event->width;
               height = configure_notify_event->height;

               pixel_count = width * height;
               pixel_count += (alignment_magnitude - (pixel_count % alignment_magnitude));
               assert(pixel_count % alignment_magnitude == 0);

               if(pixels) {
                  xcb_shm_detach(_connection, _pixmap_seg);
                  shmdt(pixels);
               }
               if(_pixmap) {
                  xcb_free_pixmap(_connection, _pixmap);
               }

               _pixmap_mem = shmget(IPC_PRIVATE, pixel_count * 4, IPC_CREAT | 0777);
               _pixmap_seg = xcb_generate_id(_connection);
               pixels = shmat(_pixmap_mem, 0, 0);

               xcb_shm_attach(_connection, _pixmap_seg, _pixmap_mem, 0);
               shmctl(_pixmap_mem, IPC_RMID, 0);

               _pixmap = xcb_generate_id(_connection);
               xcb_shm_create_pixmap(_connection, _pixmap, _window, width, height, _screen->root_depth, _pixmap_seg, 0);
               xcb_flush(_connection);
            } break;
            case XCB_EXPOSE:
            {
               CAST_DEF_EVENT(expose);
            } break;
         }
         free(e);
      }

      if(should_quit) {
         break;
      }

      // ENFORCE_READ_WRITE_ORDERING;
      work_completed = 0;
      work_index = 0;
      for(int i = 0; i < processor_count; ++i) {
         sem_post(&work_semaphore);
      }
      while(work_completed != processor_count);

      xcb_copy_area(_connection, _pixmap, _window, _gcontext, 0, 0, 0, 0, width, height);
      xcb_flush(_connection);

      struct timeval tv_end_time;
      gettimeofday(&tv_end_time, 0);

      s64 counter_elapsed = ((tv_end_time.tv_sec * 1000000) + tv_end_time.tv_usec) - ((tv_start_time.tv_sec * 1000000) + tv_start_time.tv_usec);

      f64 mspf = (f64)counter_elapsed / 1000.0;
      static f64 mspf_count;
      static u64 frame_count;
      mspf_count += mspf;
      frame_count++;
      f64 frame_count_f64 = (f64)frame_count;

      printf("%.1fms/f\n", mspf);

      tv_start_time = tv_end_time;
   }
}
