#include <stdio.h>
#include <string.h>

#define HEIGHT 7 // 根据下面定义，每个字母高度均是7行

void print_F();
void print_I();
void print_S();
void print_H();
void print_C();

int func1()
{
    print_F();
    printf("\n");
    print_I();
    printf("\n");
    print_S();
    printf("\n");
    print_H();
    printf("\n");
    print_C();
    printf("\n");

    return 0;
}

void print_F()
{
    printf("########\n");
    printf("##      \n");
    printf("##      \n");
    printf("######  \n");
    printf("##      \n");
    printf("##      \n");
    printf("##      \n");
}

void print_I()
{
    printf("####\n");
    printf(" ## \n");
    printf(" ## \n");
    printf(" ## \n");
    printf(" ## \n");
    printf(" ## \n");
    printf("####\n");
}

void print_S()
{
    printf(" ###### \n");
    printf("##    ##\n");
    printf("##      \n");
    printf(" ###### \n");
    printf("      ##\n");
    printf("##    ##\n");
    printf(" ###### \n");
}

void print_H()
{
    printf("##    ##\n");
    printf("##    ##\n");
    printf("##    ##\n");
    printf("########\n");
    printf("##    ##\n");
    printf("##    ##\n");
    printf("##    ##\n");
}

void print_C()
{
    printf(" ###### \n");
    printf("##    ##\n");
    printf("##      \n");
    printf("##      \n");
    printf("##      \n");
    printf("##    ##\n");
    printf(" ###### \n");
}

const char *letters[] = {
    "\
\########@\
\##      @\
\##      @\
\######  @\
\##      @\
\##      @\
\##      @\
",
    "\
\####@\
\ ## @\
\ ## @\
\ ## @\
\ ## @\
\ ## @\
\####@\
",
    "\
\ ###### @\
\##    ##@\
\##      @\
\ ###### @\
\      ##@\
\##    ##@\
\ ###### @\
",
    "\
\##     ##@\
\##     ##@\
\##     ##@\
\#########@\
\##     ##@\
\##     ##@\
\##     ##@\
",
    "\
\ ###### @\
\##    ##@\
\##      @\
\##      @\
\##      @\
\##    ##@\
\ ###### @\
"};

int main()
{
    int i, j;
    int gap;

    printf("请输入字母的间隔(空格数)：");
    scanf("%d", &gap);

    // 一行一行打印
    for (i = 0; i < HEIGHT; i++)
    {
        // 打印所有字母的某一行
        for (j = 0; j < 5; j++)
        {
            // k 用于迭代每一行的‘#’字符
            int k = 0;
            // 计算出该字母每一行有多少个字符
            int len = strlen(letters[j]) / HEIGHT;
            // 计算当前打印第几行
            int line = i * len;

            // @标志着该行结束
            while (letters[j][line + k] != '@')
            {
                putchar(letters[j][line + k]);
                k++;
            }

            // 打印字母间的间隔(空格)
            int temp = gap;
            while (temp--)
            {
                putchar(' ');
            }
        }
        putchar('\n');
    }

    return 0;
}