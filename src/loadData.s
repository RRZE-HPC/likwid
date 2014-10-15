.intel_syntax noprefix

.text
.globl _loadData
.type _loadData, @function
_loadData :

xor rax, rax
.align 16
1:
mov  r8,  [rsi + rax]
mov  r9,  [rsi + rax + 64]
mov  r10, [rsi + rax + 128]
mov r11,  [rsi + rax + 192]
add rax, 256
cmp rax, rdi
jb 1b

ret
.size _loadData, .-_loadData


