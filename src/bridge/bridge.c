#define _GNU_SOURCE

#include <stdio.h>
#include <stdlib.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <unistd.h>

[[noreturn]] void access_daemon_main() {
    char *argv[1] = {NULL};
    int ret = execvp("likwid-accessD", argv);
    if (ret < 0) {
        printf("Failed to start likwid-accessD daemon\n", ret);
    }
    exit(ret);
}

int create_access_daemon() {
    int pid = fork();
    if (pid == 0) access_daemon_main();
    else return pid;
}

int create_bridge_socket(int id) {
    struct sockaddr_un sock_addr;

    sock_addr.sun_family = AF_LOCAL;
    snprintf(sock_addr.sun_path, sizeof(sock_addr.sun_path), "/tmp/likwid-bridge-%d", id);

    int socket_fd = socket(AF_LOCAL, SOCK_STREAM, 0);
    if (socket_fd < 0) {
        printf("Failed to start the bridge socket\n");
        return -1;
    }

    int ret = bind(socket_fd, (const struct sockaddr *) &sock_addr, sizeof(sock_addr));
    if (ret < 0) {
        printf("Failed to bind() to the bridge socket\n");
        return -1;
    }

    ret = listen(socket_fd, 128);
    if (ret < 0) {
        printf("Failed to listen() on the bridge socket\n");
        return -1;
    }

    return socket_fd;
}

[[noreturn]] void bridge_daemon_main(int socket_fd) {
    int io_buf;
    while (1) {
        int conn_fd = accept(socket_fd, NULL, NULL);

        if (conn_fd < 0) {
            printf("Failed to accept a bridge connection\n");
            exit(-1);
        }

        long io_count = recv(conn_fd, (char *) &io_buf, sizeof(io_buf), 0);
        if (io_count != sizeof(io_buf)) {
            printf("Failed to recv from the bridge socket\n");
            exit(-1);
        }

        switch (io_buf) {
            case 1: {
                int daemon_pid = create_access_daemon();

                io_buf = daemon_pid;
                io_count = send(conn_fd, (char *) &io_buf, sizeof(io_buf), 0);
                if (io_count != sizeof(io_buf)) {
                    printf("Failed to send from the bridge socket\n");
                    close(conn_fd);
                    close(socket_fd);
                    exit(-1);
                }

                break;
            }
            default: {
                printf("Unknown bridge command: %d. Ignoring.\n", io_buf);
                close(conn_fd);
                break;
            }
        }
    }
}

int create_bridge_daemon(int id) {
    int socket_fd = create_bridge_socket(id);
    if (socket_fd < 0) return socket_fd;

    int pid = fork();
    if (pid == 0) {
        bridge_daemon_main(socket_fd);
    } else return pid;
}

int main(int argc, char *const *argv) {
    int id = getpid();
    int bridge_pid = create_bridge_daemon(id);
    if (bridge_pid < 0) return bridge_pid;

    char env_var[128];
    snprintf(env_var, sizeof(env_var), "LIKWID_BRIDGE_PATH=/tmp/likwid-bridge-%d", id);
    char *envp[2] = {env_var, NULL};

    int child_pid = fork();
    if (child_pid == 0) {
        if (argc == 1) {
            char *child_argv[2] = {"-i", NULL};
            int ret = execve("/bin/bash", child_argv, envp);
            if (ret < 0) printf("Failed to invoke bash\n");
        } else {
            const char *child_path = argv[1];
            char *const *child_argv = argv + 1;
            int ret = execvpe(child_path, child_argv, envp);
            if (ret < 0) printf("Failed to invoke %s\n", child_path);
        }
    } else {
        waitpid(child_pid, NULL, 0);
    }

    kill(bridge_pid, SIGKILL);
}
