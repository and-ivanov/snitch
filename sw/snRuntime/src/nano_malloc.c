#include "nano_malloc.h"

#include <stdlib.h>
#include <stdio.h>
#include <limits.h>

/* 
 * Memory layout:
 * [ ....segment1.... .....segment2.... ..segment3. ]
 * [ header1 .data1.. header2 ..data2.. header_last ]
 * buffer_start                            buffer_end
 * <- left                                   right ->
 *
 * Allocation strategy:
 *     Do linear search for a segment with enough data size from left to right
 *     If segment with appropriate data size is found and is free
 *         If segment is big enough (can fit additional header)
 *         then split it into two segments
 *             First (left) segment will be kept as unallocated
 *             Second (right) segment is allocated and returned
 *         Else reserve segment enitrely and return pointer to data
 *     Else return NULL
 *
 * Rationale:
 *     - Keep the design as simple as possible.
 *     - Prioritize the allocation in previously unused space,
 *       assuming total memory size is large enough,
 *       at the expense of a longer search when memory is close to exhaustion.
 *     - Keep the complexity of scope-based malloc/free pairs that use
 *     	 the inverse order of freeing as O(1).
 *     Example:
 *         void* x = malloc(100); void* y = malloc(100); void* z = malloc(100);
 *         {
 *             void* a = malloc(100); void* b = malloc(100);
 *             ...
 *             free(b); free(a);
 *         }
 *         free(z); free(y); free(x);
 */


// Segment header
typedef struct {
	// distance in bytes from the start of previous header to the start of current header
	// stores 0 for first header
	int prev;  
	// distance in bytes from the start of current header to the start of the next header
	// stores 0 for last header
	int next;  
	// boolean flag showing if data following the header is in use or available for allocation
	int is_free;  
} alloc_seg_hdr;


/*
 * Fills first and last header to form the initial layout.
 * [ header_first free_data header_last ]
 */
int alloc_init(void* buffer, int buf_size) {
	// check that there is enough space for 2 headers
	if (buf_size < 2 * sizeof(alloc_seg_hdr)) {
		return -1;  // not enough space
	}

	char* first_seg_start = buffer;  // byte-level address offsetting
	alloc_seg_hdr* first_seg_hdr = (alloc_seg_hdr*)first_seg_start;
	char* last_seg_start = first_seg_start + buf_size - sizeof(alloc_seg_hdr);
	alloc_seg_hdr* last_seg_hdr = (alloc_seg_hdr*)last_seg_start;

	first_seg_hdr->prev = 0;  // 0 -> there is no previous segment
	first_seg_hdr->next = last_seg_start - first_seg_start;
	first_seg_hdr->is_free = 1;

	last_seg_hdr->prev = last_seg_start - first_seg_start;  
	last_seg_hdr->next = 0;  // 0 -> there is no next segment
	last_seg_hdr->is_free = 0;

	return 0;  // success
}

void* alloc_malloc(void* buffer, int size) {
	if (size == 0) return NULL;
	char* seg_start = (char*)buffer;
	alloc_seg_hdr* hdr = (alloc_seg_hdr*)seg_start;
	// naive linear search for the first segment that have enough space
	while (1) {
		if (hdr->is_free && hdr->next >= size + sizeof(alloc_seg_hdr)) {
			// we found it, allocate and return
			char* buf_start = seg_start + sizeof(alloc_seg_hdr);
			int space_left = hdr->next - size - sizeof(alloc_seg_hdr);
			if (space_left < sizeof(alloc_seg_hdr)) {
				// give entire segment to this allocation
				hdr->is_free = 0;
				return buf_start;
			} else {
				// split segment into two
				int allocated_seg_size = sizeof(alloc_seg_hdr) + size;
				char* next_seg_start = seg_start + hdr->next;
				alloc_seg_hdr* next_hdr = (alloc_seg_hdr*)next_seg_start;
				char* new_seg_start = next_seg_start - allocated_seg_size;
				alloc_seg_hdr* new_hdr = (alloc_seg_hdr*)new_seg_start;
				// fill fields of the new (allocated) segment
				new_hdr->next = allocated_seg_size;
				new_hdr->prev = hdr->next - allocated_seg_size;
				new_hdr->is_free = 0;
				// adjust fields of existing shrinked (free) segment
				hdr->next -= allocated_seg_size;
				hdr->is_free = 1;
				// adjust fields of next segment
				next_hdr->prev = new_hdr->next; 
				return new_seg_start + sizeof(alloc_seg_hdr);
			}
		}
		if (hdr->next == 0) break;
		seg_start += hdr->next;
		hdr = (alloc_seg_hdr*)seg_start;
	};
	return NULL;
}

void alloc_free(void* ptr) {
	if (!ptr) return;
	char* cur_ptr = (char*)ptr - sizeof(alloc_seg_hdr);
	alloc_seg_hdr* cur_hdr = (alloc_seg_hdr*)cur_ptr;
	char* prev_ptr = cur_ptr - cur_hdr->prev;
	alloc_seg_hdr* prev_hdr = (alloc_seg_hdr*)prev_ptr;
	char* next_ptr = cur_ptr + cur_hdr->next;
	alloc_seg_hdr* next_hdr = (alloc_seg_hdr*)next_ptr;
	char* next_next_ptr = next_ptr + next_hdr->next;
	alloc_seg_hdr* next_next_hdr = (alloc_seg_hdr*)next_next_ptr;

	cur_hdr->is_free = 1;
	if (next_hdr->is_free) {
		// merge current with next
		cur_hdr->next = cur_hdr->next + next_hdr->next;
		next_next_hdr->prev = cur_hdr->next;
		next_ptr = next_next_ptr;
		next_hdr = next_next_hdr;
	}
	if (prev_hdr->is_free && prev_hdr != cur_hdr) {
		// merge current with previous
		prev_hdr->next = prev_hdr->next + cur_hdr->next;
		next_hdr->prev = prev_hdr->next;
	}
}
