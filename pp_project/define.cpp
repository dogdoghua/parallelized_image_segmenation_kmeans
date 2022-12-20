#include "define.h"

FILE *f;
uint8_t *image, *origin_image, *knn_image;
int img_width, img_height, img_channels, img_size;

unsigned int get_offset(int y, int x, int c) {
    return img_channels * (y * img_width + x) + c;
}