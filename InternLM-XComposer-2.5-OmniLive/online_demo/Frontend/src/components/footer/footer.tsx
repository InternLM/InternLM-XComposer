import style from './footer.module.less';
import Logo from '@/assets/svg/logo.svg';

const Footer = () => {
    return (
        <div className={style.footer}>
            <img className={style.footerLogo} src={Logo} alt="" />
            <div className={style.footerDesc}>实时和我对话交流</div>
        </div>
    );
};

export default Footer;
