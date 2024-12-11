import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import basicSsl from '@vitejs/plugin-basic-ssl';
import path from 'path';

// https://vitejs.dev/config/
export default ({ mode }) => {
    return defineConfig({
        plugins: [
            react({
                babel: {
                    plugins: [
                        '@babel/plugin-proposal-optional-chaining', // 兼容老版本浏览器的语法解译
                    ],
                },
            }),
            ...(mode === 'https' ? [basicSsl()] : []),
        ],
        resolve: {
            alias: {
                '@': path.resolve(__dirname, 'src'),
            },
        },
        css: {
            modules: {
                localsConvention: 'camelCase',
            },
        },
        server: {
            port: 8081,
            proxy: {
                '/rtc': {
                    target: 'http://localhost:1985',
                    changeOrigin: true,
                },
            },
        },
    });
};
