#include <stdio.h>
#include <stdlib.h>

/*
int main(int argc, char *argv[])
{
    int i;
    int sum = 0;

    for (i = 0; i < argc; i++)
    {
        sum += atoi(argv[i]);
    }

    printf("sum = %d\n", sum);

    return 0;
}
*/

/*
int main(int argc, char *argv[])
{
    int result = 0;

    while (argc-- != 1)
    {
        result += atoi(argv[argc]);
    }

    printf("sum = %d\n", result);

    return 0;
}
*/

void func1()
{
    char *array[5] = {"FishC","Five","Star","Good","WOW"};
    char *(*p)[5] = &array;
    int i,j;

    for (i = 0; i < 5; i++)
    {
        for (j = 0; (*p)[i][j] != '\0'; j++)
        {
            printf("%c ", (*p)[i][j]);
        }
        printf("\n");
    }
}

void func2()
{
    char *array[5] = {"FishC", "Five", "Star", "Good", "WOW"};
    char *(*p)[5] = &array;
    int i, j;

    for (i = 0; i < 5; i++)
    {
        for (j = 0; j < 5; j++)
        {
            if ((*p)[j][i] == '\0')
            {
                break;
            }
            printf("%c ", (*p)[j][i]);
        }
        printf("\n");
    }
}

int main()
{
    // func1();
    func2();

    return 0;
}
