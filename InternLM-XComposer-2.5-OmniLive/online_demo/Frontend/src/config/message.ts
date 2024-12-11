export enum SOCKET_MESSAGE_MAP {
    FIRST_MESSAGE = '@@socket_ready',
    VOICE_START = '@@voice_start',
    VOICE_END = '@@voice_end',
    SOCKET_FORCE_CLOSED = '@@socket_force_closed',
    INTERRUPT_VOICE = '@@interrupt',
}
