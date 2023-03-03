#pragma once

int alloc_init(void* buffer, int buf_size);
void* alloc_malloc(void* buffer, int size);
void alloc_free(void* ptr);