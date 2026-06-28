#include <stdio.h>

int generate_config(void);

int main(void)
{
    printf("=========================================\n");
    printf("CONFIG GENERATOR UNIT TEST\n");
    printf("=========================================\n\n");

    /* Generate encrypted configuration */

    if(generate_config() != 0)
    {
        printf("FAIL : Could not generate config.bin\n");
        return -1;
    }

    printf("PASS : config.bin generated successfully\n");

    /* Verify file exists */

    FILE *fp = fopen("config.bin", "rb");

    if(fp == NULL)
    {
        printf("FAIL : config.bin not found\n");
        return -1;
    }

    printf("PASS : config.bin exists\n");

    /* Verify file size */

    fseek(fp, 0, SEEK_END);

    long file_size = ftell(fp);

    fclose(fp);

    printf("File Size : %ld bytes\n", file_size);

    if(file_size > 0)
        printf("PASS : File is not empty\n");
    else
        printf("FAIL : File is empty\n");

    printf("\nGenerator Unit Test Completed\n");

    return 0;
}
