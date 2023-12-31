CDF  �   
      time          *   Conventions       ACDD-1.3, Spase v2.2.3     title         /DSCOVR Magnetometer Level 2 One Minute Averages    id        Doe_m1m_dscovr_s20230325000000_e20230325235959_p20230326021558_pub.nc   naming_authority      gov.noaa.swpc      program       DSCOVR     summary       }Interplanetary magnetic field observations collected from magnetometer on DSCOVR satellite - 1-minute average of Level 1 data      keywords      _NumericalData.ObservedRegion.Heliosphere.NearEarth, NumericalData.MeasurementType.MagneticField    keywords_vocabulary       Spase v2.2.2   
references        �; DSCOVR TIME SERIES DATA AVERAGES ALGORITHM THEORETICAL BASIS DOCUMENT, v2.4; GSE TO GSM COORDINATE TRANSFORMATION ALGORITHM THEORETICAL BASIS DOCUMENT v2.1      metadata_link         �http://www.ngdc.noaa.gov/docucomp/page?xml=NOAA/NESDIS/NGDC/STP/Space_Weather/iso/xml/satellite-systems_dscovr.xml&view=getDataView&header=none    license       Spase.Access Rights.Open   institution       NOAA   source        DSCOVR Magnetometer Level 1    platform      'Deep Space Climate Observatory (DSCOVR)    
instrument        +boom-mounted triaxial fluxgate magnetometer    history       ,DSCOVR real-time telemetry processing system   	algorithm         FDSCOVR MAGNETOMETER LEVEL 1B DATA ALGORITHM THEORETICAL BASIS DOCUMENT     algorithmVersion      B      algorithmDate         
2015-10-15     processing_level      Level 2    processing_level_description      11-minute average using Hodges-Lehmann M-estimator      date_created      2023-03-26T02:15:58.660Z   date_calibration_data_updated         2023-01-28T00:00:00.000Z   time_coverage_duration        P01D   time_coverage_start       2023-03-25T00:00:00.000Z   time_coverage_end         2023-03-25T23:59:59.000Z   time_coverage_resolution      PT1M   creator_name      Doug Biesecker     creator_type      person     creator_institution       DOC/NOAA/NWS/NCEP/SWPC     creator_email         doug.biesecker@noaa.gov    creator_url       http://www.swpc.noaa.gov/      publisher_name         National Geophysical Data Center   publisher_type        institution    publisher_institution         DOC/NOAA/NESDIS/NGDC   publisher_email       william.rowland@noaa.gov   publisher_url          http://www.ngdc.noaa.gov/dscovr/   records_maximum         �   records_present         �   records_data        �   records_fill             records_missing                    time                description       "date and time for each observation     
short_name        time   C_format      %.13g      units         'milliseconds since 1970-01-01T00:00:00Z    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   standard_name         time   calendar      	gregorian           7   sample_count                description       /number of full resolution measurements averaged    
short_name        sample_count   C_format      %d     units         samples    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	valid_min                	valid_max           �        7   measurement_mode                description       7measurement range selection mode (0 = auto, 1 = manual)    
short_name        mode   C_format      %1d    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	valid_min                	valid_max                    7   measurement_range                   description       5measurement range (~4x sensitivity increase per step)      
short_name        range      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	valid_min                	valid_max                    7   bt               	   description       )Interplanetary Magnetic Field strength Bt      
short_name        bt     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         )bt_interplanetary_magnetic_field_strength      	valid_min                	valid_max                    7    bx_gse               
   description       \Interplanetary Magnetic Field strength Bx component in Geocentric Solar Ecliptic coordinates   
short_name        bx_gse     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bx_interplanetary_magnetic_field_strength_gse      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         7$   by_gse               
   description       \Interplanetary Magnetic Field strength By component in Geocentric Solar Ecliptic coordinates   
short_name        by_gse     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -by_interplanetary_magnetic_field_strength_gse      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         7(   bz_gse               
   description       \Interplanetary Magnetic Field strength Bz component in Geocentric Solar Ecliptic coordinates   
short_name        bz_gse     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bz_interplanetary_magnetic_field_strength_gse      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         7,   	theta_gse                	   description       RInterplanetary Magnetic Field clock angle in Geocentric Solar Ecliptic coordinates     
short_name        	theta_gse      C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min         ����   	valid_max            Z   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         70   phi_gse              	   description       RInterplanetary Magnetic Field polar angle in Geocentric Solar Ecliptic coordinates     
short_name        phi_gse    C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min                	valid_max           h   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         74   bx_gsm               
   description       bInterplanetary Magnetic Field strength Bx component in Geocentric Solar Magnetospheric coordinates     
short_name        bx_gsm     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bx_interplanetary_magnetic_field_strength_gsm      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         78   by_gsm               
   description       bInterplanetary Magnetic Field strength By component in Geocentric Solar Magnetospheric coordinates     
short_name        by_gsm     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -by_interplanetary_magnetic_field_strength_gsm      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7<   bz_gsm               
   description       bInterplanetary Magnetic Field strength Bz component in Geocentric Solar Magnetospheric coordinates     
short_name        bz_gsm     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bz_interplanetary_magnetic_field_strength_gsm      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7@   	theta_gsm                	   description       XInterplanetary Magnetic Field clock angle in Geocentric Solar Magnetospheric coordinates   
short_name        	theta_gsm      C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min         ����   	valid_max            Z   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7D   phi_gsm              	   description       XInterplanetary Magnetic Field polar angle in Geocentric Solar Magnetospheric coordinates   
short_name        phi_gsm    C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min                	valid_max           h   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7H   backfill_flag                   description       �One or more measurements were backfilled from the spacecraft recorder and therefore were not available to forecasters in real-time     
short_name        backfill_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         backfilled_data_flag   	valid_min                	valid_max                    7L   future_packet_time_flag                 description       rOne or more measurements were extracted from a packet whose timestamp was in the future at the point of processing     
short_name        future_packet_time_flag    C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         packet_time_in_future_flag     	valid_min                	valid_max                    7P   old_packet_time_flag                description       }One or more measurements were extracted from a packet whose timestamp was older than the threshold at the point of processing      
short_name        old_packet_time_flag   C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         %packet_time_older_than_threshold_flag      	valid_min                	valid_max                    7T   	fill_flag                   description       Fill   
short_name        	fill_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         	fill_flag      	valid_min                	valid_max                    7X   possible_saturation_flag                description       �Possible magnetometer saturation based on a measurement range smaller than the next packet's range or by the mag being in manual range mode.   
short_name        possible_saturation_flag   C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         %possible_magnetometer_saturation_flag      	valid_min                	valid_max                    7\   calibration_mode_flag                   description       Instrument in calibration mode     
short_name        calibration_mode_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         calibration_mode_flag      	valid_min                	valid_max                    7`   maneuver_flag                   description       4AOCS non-science mode (spacecraft maneuver/safehold)   
short_name        maneuver_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         /AOCS_non_science_mode_maneuver_or_safehold_flag    	valid_min                	valid_max                    7d   low_sample_count_flag                   description       $Average sample count below threshold   
short_name        low_sample_count_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         )average_sample_count_below_threshold_flag      	valid_min                	valid_max                    7h   overall_quality                 description       ;Overall sample quality (0 = normal, 1 = suspect, 2 = error)    
short_name        overall_quality    C_format      %d     units         n/a    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale        7lBxqa	�  �          A�@��H����@���B G�C�'�@��H�ƸR@33Ar{C�}q                                    Bxqaf  �          A\)@�����Q�@u�A�Q�C���@����Ϯ>�
=@<��C��
                                    Bxqa'  
�          A33@�  ����@q�Aۙ�C���@�  ��  >�Q�@"�\C��f                                    Bxqa5�  
�          @��@������@��A�{C���@�����
�����
�RC���                                    BxqaDX  
�          A z�@�33���R@O\)A�
=C�q�@�33��녾��R�(�C���                                    BxqaR�  F          @��@�(���\)@+�A�33C�
@�(��Ϯ�n{��Q�C���                                    Bxqaa�  �          @��@~�R����@ ��An�RC��@~�R��\)��(��K�
C��f                                    BxqapJ  
�          @���@i���θR?�z�AI��C��
@i����33�33�z{C��                                    Bxqa~�  �          @�ff@hQ����@  A���C��f@hQ����
�\�6ffC�b�                                    Bxqa��  
(          @�{@Fff��G�@'�A��C���@Fff�����
���C��                                    Bxqa�<  �          @�z�@1G����
@I��A��HC���@1G���=q�=p�����C���                                    Bxqa��  
~          @�{@=p��ƸR@Z=qA��C���@=p���=q��
=�I��C�^�                                    Bxqa��  �          @��@B�\��ff@z=qA���C���@B�\���
>\)?��
C��3                                    Bxqa�.  �          @�
=@4z����H@��BC��f@4z����>��@C�
C�                                    Bxqa��  T          @���@\�����
@�  A�=qC��H@\����z�>���@<��C�H�                                    Bxqa�z  "          @��
@������@dz�A؏\C��3@����33=�\)>��HC�^�                                    Bxqa�   �          @�z�@�����H@XQ�A�G�C�)@����G���G��Q�C��q                                    Bxqb�  T          A (�@�G����R@h��A�  C�s3@�G�����=#�
>���C�+�                                    Bxqbl  "          @��@������@S�
A�{C�xR@������\)��C��
                                    Bxqb   "          @�
=@�����H@C33A��HC���@����=q���r�\C���                                    Bxqb.�  �          @��@��\��G�@mp�A���C�.@��\��p�>L��?��HC���                                    Bxqb=^  �          @�{@vff��33@�{B��C�~�@vff�љ�?W
=@�G�C�5�                                    BxqbL  T          @���@u���H@UA�{C���@u��
=��=q��C��                                    BxqbZ�  
�          @��@�\)��G�@�ArffC�Ff@�\)��ff��p��/\)C��                                    BxqbiP  �          @���@��R��@8Q�A�{C�@��R��33����
=C��=                                    Bxqbw�  �          @�  @�����H?�Ag�
C�"�@����\)��(��0(�C��3                                    Bxqb��  "          @�=q@�33��(�?���AX  C��@�33�����Q��)��C��3                                    Bxqb�B  �          @��@�\)���?�ff@�p�C�'�@�\)��p��
=q��C��R                                    Bxqb��  �          @���@�(����
?��RA�
C��)@�(���z���
�v�RC�o\                                    Bxqb��  �          @�Q�@��H��@>{A���C���@��H��z���e�C��                                    Bxqb�4  "          @�
=@�=q����@G
=A��C��@�=q�ʏ\��\)��C�'�                                    Bxqb��  S          @�
=@����\?�{A  C�
@���G��	����G�C���                                    Bxqbހ  
�          @�p�@��H���@�A�Q�C���@��H��(�����\C���                                    Bxqb�&  T          @��R@����
@K�A�\)C�P�@��θR��=q�G�C�w
                                    Bxqb��  �          @�\)@������@5�A��C�p�@�����  �\)����C��                                    Bxqc
r  
�          @�ff@�{����@�\Aup�C���@�{��
=�����C�\                                    Bxqc  �          @�ff@����  @ ��A�=qC��R@����  �E����\C���                                    Bxqc'�  �          @��
@�Q����H?�
=Av�RC��@�Q���녿�
=��C���                                    Bxqc6d  
�          @�(�@�Q����?Y��@�p�C���@�Q�������\)C���                                    BxqcE
  T          @�ff@��\��p�?.{@��HC���@��\��z��"�\���HC��\                                    BxqcS�  T          @�  @�G�����?�\)A_�C���@�G����Ϳ��H�-C���                                    BxqcbV  �          @��\@���
=@!�A�
=C��@���ff�\(���33C��=                                    Bxqcp�  T          @�  @�(����
@	��A���C�u�@�(����������\C��H                                    Bxqc�  "          @�\)@�z�����@$z�A�=qC��
@�z��ȣ׿Y���˅C�|)                                    Bxqc�H  �          @��@xQ���G�@`��A��C��=@xQ��أ׽u��
=C���                                    Bxqc��  �          @�
=@�����@5A���C�(�@������ÿ333��C��H                                    Bxqc��  "          @�  @tz���
=@FffA�33C�L�@tz��ָR�   �j�HC���                                    Bxqc�:  "          @�G�@�G���{@?\)A���C�)@�G���(��z���  C��\                                    Bxqc��  �          @��@vff����@7
=A�C�>�@vff���Ϳ=p�����C��                                    Bxqc׆  �          @�@y�����@<��A���C��@y���Ϯ�
=q���C�z�                                    Bxqc�,  �          @�{@������@/\)A��HC��=@�����p��=p�����C�p�                                    Bxqc��  "          @�@�����@dz�A�{C��@����G�?J=q@��C�5�                                    Bxqdx  �          @�=q@�(���=q@Y��A�  C�o\@�(���  ?B�\@���C��R                                    Bxqd  1          @��@��\��{@��RB�RC�&f@��\��\)?�G�A�C���                                    Bxqd �  
}          @�G�@�Q����@s33A�C���@�Q���(�?0��@���C�c�                                    Bxqd/j  
�          @��@���(�@�
=B��C��H@�����\@7�A�
=C��f                                    Bxqd>  "          @��
@��H����@�  B9�C�` @��H�y��@�=qB��C�E                                    BxqdL�  T          @��@���\)@���BDz�C��)@���[�@���B(�C�j=                                    Bxqd[\  
�          @���@��\��  @��RBAC�]q@��\��Q�@��\B�C�L�                                    Bxqdj  
�          @�
=@�
=���@��B%Q�C��)@�
=�I��@�  A��C���                                    Bxqdx�  "          @�
=@�ff��(�@��B  C��@�ff�k�@Mp�A�33C�                                      Bxqd�N  �          @��@�{��G�@��
B(�C��@�{�:=q@{�A��
C��                                    Bxqd��  
�          @���@�=q�'�@�G�B�C���@�=q��
=@33A�33C�=q                                    Bxqd��  �          @�G�@�ff���
@��B��C��=@�ff�|��@C33A�\)C�7
                                    Bxqd�@  �          @���@�  ��(�@�B�C���@�  ����@Tz�A�{C�z�                                    Bxqd��  
�          @��H@�Q��%@�{Bp�C��@�Q�����@��A��
C��                                    BxqdЌ  T          @��@�Q���@���B ��C��@�Q���=q@HQ�A��HC�+�                                    Bxqd�2  �          @��@�ff�C�
@��\B G�C���@�ff��  @0  A�G�C�]q                                    Bxqd��  
�          @��@���E�@��HB\)C�+�@������@"�\A�{C�9�                                    Bxqd�~  �          @��R@���e�@�ffB#{C��@����Q�@%A��C�3                                    Bxqe$  �          @�\)@���z�@���B	��C�g�@���G�?�p�A/�C��                                    Bxqe�  
(          Ap�@�\)��G�@<(�A���C�J=@�\)��{�(����C��=                                    Bxqe(p  �          A ��@�  ��Q�@qG�A�(�C�@�  �θR?   @eC�(�                                    Bxqe7  
�          A=q@�����p�@��B �C�@����׮?�A   C��                                    BxqeE�  T          A
=@�G���{@�  A�  C�e@�G��߮?��@�(�C��f                                    BxqeTb  
�          A��@��H����@7�A�C���@��H�޸R�W
=��p�C�h�                                    Bxqec  �          A��@�  �ə�@5A���C�5�@�  �ۅ�Q�����C��                                    Bxqeq�  �          A
ff@������@HQ�A��
C���@�����(���{���C���                                    Bxqe�T  �          Ap�@���  @eA��C�Q�@��أ�=�G�?:�HC���                                    Bxqe��  �          A{@�
=���
@q�A��C�@�
=�љ�>�@E�C�+�                                    Bxqe��  T          A{@������@e�A��C��{@�����Q�=�G�?=p�C�<)                                    Bxqe�F  �          A�@��H��G�@c�
A��HC�e@��H��Q�u�\C�K�                                    Bxqe��  �          A\)@�(����H@p  A�(�C��q@�(�����=�Q�?
=qC�}q                                    Bxqeɒ  "          A{@�ff��\)@��HA��C�g�@�ff��{?h��@�(�C��                                    Bxqe�8  �          A=q@��\��Q�@�p�A��C���@��\���
?@  @�=qC���                                    Bxqe��  T          A\)@���ff@�Q�A㙚C���@���\?=p�@��RC��q                                    Bxqe��  T          A�@�����
=@��RA�(�C���@�����\)?��\@љ�C�AH                                    Bxqf*  
�          A�
@�=q��z�@�G�B ��C��R@�=q���?�{A%�C��3                                    Bxqf�  
�          A�R@�����@��B{C��H@���Ϯ?��A9G�C��                                    Bxqf!v  "          A�
@�{��p�@�
=BC�P�@�{��?�A0  C��H                                    Bxqf0  "          A  @�����R@U�A�Q�C��
@�����
�u�ǮC���                                    Bxqf>�  "          A
=@�\)���
@
�HAb{C��H@�\)��׿���*{C�W
                                    BxqfMh  �          A��@�p���{@UA�p�C�9�@�p����þ�z��{C�o\                                    Bxqf\  "          A��@�ff��G�@N{A�  C�R@�ff�ᙚ��G��2�\C�z�                                    Bxqfj�  �          A
�R@��R��>L��?���C�}q@��R��  �g
=����C�=q                                    BxqfyZ  "          A
�\@vff���>��
@
=C�9�@vff��=q�j�H����C���                                    Bxqf�   "          A��@Q����\���s�
C�` @Q���ff��
=���C�|)                                    Bxqf��  T          Az�@xQ���\�.{����C���@xQ���{��  ��  C��                                    Bxqf�L  �          A  @o\)��33�8Q쿙��C��@o\)�θR������ffC�f                                    Bxqf��  "          A(�@,��� Q�Q����C�J=@,���Ϯ������C�G�                                    Bxqf  T          A��@:=q�   �B�\��33C��@:=q��  ���R��HC��                                    Bxqf�>  �          A
=@c�
��(�����޸RC���@c�
��{���H��ffC�w
                                    Bxqf��  "          A�R@�33���H?�(�AffC�R@�33�\�
=q�n=qC���                                    Bxqf�  T          Aff@�p����?�(�A!�C�<)@�p��Ӆ���qp�C��                                     Bxqf�0  �          A	p�@r�\��<#�
=uC�XR@r�\��Q��s�
��z�C�)                                    Bxqg�  "          Az�@'
=�33�޸R�6�HC���@'
=��{��z�� �C�j=                                    Bxqg|  T          A�@-p��{�ff�\Q�C�8R@-p���ff��p��(�C�=q                                    Bxqg)"  "          A�H@7
=��׿ٙ��/�C���@7
=�ə������  C�:�                                    Bxqg7�  
�          A�@<���G������!p�C��)@<����������{C�Z�                                    BxqgFn  �          A�
@?\)�z����7�C��@?\)��  ��
=�{C�˅                                    BxqgU  
�          Aff@N{�
=�������C��)@N{��������(�C�W
                                    Bxqgc�  "          AQ�@j�H�����U�C�f@j�H���H���H���RC��                                    Bxqgr`  T          A�
@p����\��\)��G�C�^�@p����p����H����C�C�                                    Bxqg�  �          A��@�33� ��>B�\?�Q�C�~�@�33����xQ���p�C�{                                    Bxqg��  
�          A@������?�R@���C�s3@�����  �Vff��p�C���                                    Bxqg�R  
�          A��@���G�?�
=@��C��\@���� ����
=C���                                    Bxqg��  T          AG�@/\)�  �����"{C��\@/\)�����z��C�W
                                    Bxqg��  "          A��@S33�ff���<(�C��R@S33��G������33C���                                    Bxqg�D  
�          A�@L���Q���AG�C�N@L����z�������C�
                                    Bxqg��  �          A{@�G���R���
��C�%@�G���G������
=C���                                    Bxqg�  "          A�@��
��  ?��@���C��{@��
���<�����C���                                    Bxqg�6  �          A(�@�33��?�
=@���C�u�@�33��\�.{��z�C�4{                                    Bxqh�  �          A\)@�Q���=q?�ffA�RC��q@�Q���\���u�C�*=                                    Bxqh�  
�          A�@��
��\?�  @ʏ\C�^�@��
��=q�:=q���C�H�                                    Bxqh"(  
�          A\)@�����?Tz�@��C���@��ڏ\�>{���C���                                    Bxqh0�  T          A��@��H��?�
=A�C�P�@��H���� ����z�C��3                                    Bxqh?t  �          A{@�����z�@$z�A�p�C�� @�����\)�����RC�Ff                                    BxqhN  
�          A=q@�33���H��\)��\C�
=@�33��\)�x���ʏ\C�Q�                                    Bxqh\�  T          A�@��
�=q���5C��=@��
��Q���(���C���                                    Bxqhkf  T          A  @�����(����W
=C�n@�����z��l(���ffC�o\                                    Bxqhz  
�          A��@�G���녾�  ����C�E@�G���Q��mp���z�C���                                    Bxqh��  "          Aff@�����
����˅C���@�����H�g���p�C��3                                    Bxqh�X  "          A=q@ָR��\)����8Q�C��q@ָR�����g
=���C���                                    Bxqh��  
�          A(�@�G��ۅ��G��.{C��@�G�����^�R���\C��\                                    Bxqh��  �          A�@���Ϯ���Ϳ�RC���@�����H�R�\����C��)                                    Bxqh�J  �          A�
@�(�����?L��@��C���@�(���Q��1G����\C���                                    Bxqh��  �          A�\@�Q���p�?c�
@��C���@�Q����0  ��33C���                                    Bxqh��  "          A ��@�G���33?O\)@��C�c�@�G��ҏ\�2�\���C�z�                                    Bxqh�<  �          A
=@θR��?�
=A7�
C��H@θR��\)��p��$z�C�e                                    Bxqh��  �          A#33@���\@�ffB��C�Z�@����@��AG�C�*=                                    Bxqi�  �          A#
=@�����  @��B{C��@�����R@p�AH��C�f                                    Bxqi.  �          A"ff@�ff��  @��\Bp�C��\@�ff���@G�A8(�C��f                                    Bxqi)�  
�          A"{@�z���33@�\)B�HC��)@�z��
=?��AQ�C�G�                                    Bxqi8z  
Z          A�@�����@�(�AϮC�G�@�������?L��@�=qC�u�                                    BxqiG   �          A
=@�����ff@��HAυC��@������?aG�@��C�&f                                    BxqiU�  �          A�@���Q�@���A�{C��q@���(�>u?��C�y�                                    Bxqidl  �          Az�@��H��  @|��A���C�j=@��H��=q>#�
?k�C�\)                                    Bxqis  �          A�
@�\)��
=@�33A�\)C�8R@�\)���>��R?�=qC��                                    Bxqi��  
�          A
=@�
=��G�@~�RA�Q�C�g�@�
=���
>.{?��C�aH                                    Bxqi�^  
�          Aff@�����ff@/\)A�ffC�� @�������#�
���
C���                                    Bxqi�  �          AG�@����Q�@\��A���C��R@���ȣ�>�@0��C�c�                                    Bxqi��  �          Ap�@�(���33@\(�A�{C�J=@�(�����>.{?��
C�,�                                    Bxqi�P  �          A(�@�Q����
@c�
A���C�~�@�Q���׽�Q�\)C��
                                    Bxqi��  
�          Az�@�ff��z�@�p�A�C�޸@�ff��?\)@VffC�j=                                    Bxqiٜ  
�          A��@޸R���?��A�
C�<)@޸R��=q��\�-�C�u�                                    Bxqi�B  T          A�@�=q���H?\)@W�C��R@�=q��{��K�C���                                    Bxqi��  
�          A�
@�����?��H@��
C�O\@������H�Ǯ��C���                                    Bxqj�  T          A@���=q@=qAi�C�t{@���G����\(�C�S3                                    Bxqj4  �          A@�(���ff@	��APQ�C�'�@�(����ÿ\(���(�C�\)                                    Bxqj"�  
(          A��@�  ���?˅AC���@�  ��
=��
=��33C�e                                    Bxqj1�  T          A�R@�(���G�@_\)A�G�C��@�(���(�?B�\@�\)C�0�                                    Bxqj@&  	�          A�\@���  @b�\A���C�#�@��\?333@�C�u�                                    BxqjN�  �          A
=@�(���
=@g
=A�33C�B�@�(�����?��@P  C��H                                    Bxqj]r  �          A�R@�33���@9��A���C�q�@�33�ȣ׽��5C��                                    Bxqjl  �          A�R@�G���  @%Ay�C�޸@�G���G�����333C���                                    Bxqjz�  T          A�@�z��ʏ\@�AH  C��R@�z��љ�������ffC�:�                                    Bxqj�d  "          A33@�\)�θR?��A333C��
@�\)�ҏ\��
=��
C�W
                                    Bxqj�
  �          A��@�{��p�@$z�As�C���@�{���
�L����{C���                                    Bxqj��  �          Ap�@�Q���{@C33A�ffC�G�@�Q���zᾊ=q�ǮC��q                                    Bxqj�V  �          A{@�\)��{@Mp�A�C�7
@�\)��
=���:�HC���                                    Bxqj��  �          A�@˅��(�?�
=A:=qC�aH@˅��
=�����33C�1�                                    BxqjҢ  
�          Az�@�z��ҏ\@8��A�\)C���@�z�����z��Y��C�l�                                    Bxqj�H  
Z          A=q@�=q��@E�A�33C�j=@�=q���;aG����
C��
                                    Bxqj��  T          A�@޸R���H@<��A�\)C��{@޸R��\)�����C�s3                                    Bxqj��  T          A�@�G���(�@5�A��RC�*=@�G��ə��.{��G�C��
                                    Bxqk:  �          A�H@������
@>{A��\C�C�@�����(�=��
>��HC�k�                                    Bxqk�  �          A=q@ٙ�����@
=Ai�C�>�@ٙ���p��B�\���C�S3                                    Bxqk*�  T          A
=@�=q��p�@5A��HC�c�@�=q����=�\)>�(�C���                                    Bxqk9,  	�          A�@陚��ff@!G�Aw
=C�� @陚��  ��33�
=C�N                                    BxqkG�  
�          A  @�Q���G�@Q�A��C��R@�Q�����?#�
@uC�9�                                    BxqkVx  T          A  @������@@  A�G�C��=@�����?   @B�\C���                                    Bxqke             A�A ����(�@.�RA�ffC�l�A ����>��@7�C�C�                                    Bxqks�  
�          A�A{�j=q@Ak
=C�� A{���\>�33@
=qC���                                    Bxqk�j  �          A
=Aff���@5A�ffC���Aff� ��?��A3
=C�S3                                    Bxqk�  
Z          A�RAff�5@\��A�33C��)Aff�\)@.{A�(�C��                                    Bxqk��  �          Aff@����ff@6ffA�  C��{@�����R>W
=?�  C���                                    Bxqk�\  "          A��A���j=q@7
=A��RC�nA����=q?Q�@�33C��3                                    Bxqk�  T          Az�A�
��Q�@\��A�{C�t{A�
�8Q�@Q�Ab�RC�q                                    Bxqk˨  �          A�A\)?��@�  A��
@g�A\)���R@���AîC�"�                                    Bxqk�N  T          A  A(�>��
@���A�33@z�A(���(�@}p�A�G�C�p�                                    Bxqk��  
�          A�A\)�\)@p��A�  C��\A\)��\)@P��A���C�q                                    Bxqk��  "          AG�@����
=?s33@�z�C���@�����ÿٙ��"�RC�\                                    Bxql@  r          A{@�Q���Q�>��?n{C��@�Q����H�8Q����C���                                    Bxql�  �          AQ�@�(���Q�#�
���\C�aH@�(���p��s�
��=qC���                                    Bxql#�  T          Ap�@����ۅ�^�R���C�o\@������|(���G�C��                                    Bxql22  �          A	p�@��
���!G���
=C�h�@��
��33�����ߙ�C�|)                                    Bxql@�  "          A
ff@}p���33����h��C���@}p��Ϯ�\)��\)C���                                    BxqlO~  "          A
=@o\)��Q쿆ff��ffC��H@o\)������  ��=qC�(�                                    Bxql^$  
�          A��@c33��\)��=q���C���@c33���H��(���C�#�                                    Bxqll�  �          A	��@`����������C�G�@`����
=��{�(�C��
                                    Bxql{p  �          A��@��\���8Q����\C�G�@��\��=q���\�㙚C�ff                                    Bxql�  �          A�
@�
=��  �h����{C�/\@�
=���H�y����z�C��R                                    Bxql��  "          A	�@�Q����ÿ�  �p�C�o\@�Q���G��w����C���                                    Bxql�b  �          A
ff@���p���\)��
=C�L�@������e��Q�C�l�                                    Bxql�  �          A�@���������#
=C�ff@������
�vff�љ�C��                                    BxqlĮ  �          A\)@ٙ����\�$z���p�C���@ٙ��p���z���ffC��                                    Bxql�T  �          A(�@�(��0  �n{��p�C��@�(��@  ������C��                                    Bxql��  �          A\)@�z�����?�
=A��C�L�@�z���Q�s33��z�C��q                                    Bxql�  
�          A(�@�����
?h��@�
=C���@����
=��G���C�Y�                                    Bxql�F  �          A�@��H��  >���@*=qC�c�@��H��z��z��K\)C�c�                                    Bxqm�  
�          A
�\@�Q����H>�ff@@  C�}q@�Q�����  �s
=C��{                                    Bxqm�  �          Az�@���˅?���A��C�S3@�����ÿ�
=�0��C��H                                    Bxqm+8  
�          A(�@�������HQ�C�*=@�������U����
C�XR                                    Bxqm9�  �          A@��H���R=u>\C�t{@��H��33�(����33C��)                                    BxqmH�  
�          Az�@�����=q@(��A�G�C�g�@�����zᾅ��У�C�3                                    BxqmW*  �          A�@�G��Ϯ@33Au��C��)@�G��ڏ\�^�R���C��f                                    Bxqme�  
�          A33@����Ӆ?�(�AQG�C�^�@�����G����H� (�C�                                      Bxqmtv  �          A
{@��
���?�AD��C���@��
�θR��
=���
C���                                    Bxqm�  �          A=q@��8��?޸RAAG�C���@��W
=>��?��C��                                    Bxqm��  "          A�H@�R�R�\��G��J=qC�q@�R�8�ÿ����1C�k�                                    Bxqm�h  "          A  @�(��@  ?!G�@��
C�H�@�(��=p��G���{C�g�                                    Bxqm�  
�          A  @�Q��,��?�\@aG�C�h�@�Q��(�ÿ@  ��  C���                                    Bxqm��  "          A��@����(Q�?Tz�@�G�C��f@����.�R��(��>�RC�W
                                    Bxqm�Z  
�          AQ�@��R���?��
AG�
C���@��R�4z�?
=@��\C���                                    Bxqm�   T          A��@�(��*�H@333A���C���@�(��h��?���AC�y�                                    Bxqm�  "          A@�
=�$z�@��A�ffC�4{@�
=�W
=?�  @�z�C���                                    Bxqm�L  �          @�\)@���l��?�ffAP��C��H@��������aG�C�s3                                    Bxqn�  
(          A (�@������R@�Av{C�P�@�����{���
���C��                                    Bxqn�  �          A ��@���(�?޸RAQC��3@���33�Q�����C�c�                                    Bxqn$>  
�          @��@8����p�?У�A>=qC��
@8��������H�G�
C��q                                    Bxqn2�  
          @�@Tz���  ?�z�AA��C���@Tz���Q�����:=qC��=                                    BxqnA�            @��@Z�H��{?�33A@Q�C�!H@Z�H��ff��=q�8(�C��                                    BxqnP0  "          @�@a���p�?�ffA4z�C���@a���z��33�@��C���                                    Bxqn^�  T          @���@@����?k�@ָRC�]q@@����33����z�C�ٚ                                    Bxqnm|  "          A ��@�����@
=As�C�s3@�����G������  C���                                    Bxqn|"  "          @�{@������?�  AMG�C���@�����R�s33��p�C�L�                                    Bxqn��  �          @�(�@�=q��
=?�G�AO33C�q�@�=q��zῃ�
��C�                                    Bxqn�n  T          @�=q@����ff?��\A�\C�� @��������H�,(�C���                                    Bxqn�  "          @��R@��   ��
=�K�C�Ff@���\)��  ���C��3                                    Bxqn��  
�          @�\)@׮@�?�{A,  A���@׮?��@  A�Q�AM��                                    Bxqn�`  T          @�=q@�@
�H@��A��A��@�?xQ�@>�RA�z�@�\)                                    Bxqn�  �          @�G�@ᙚ@(�@p�A�33A�@ᙚ?fff@J=qA�@�G�                                    Bxqn�  �          @�ff@�\?�=q@Q�A�G�AI��@�\>Ǯ@5�A�(�@K�                                    Bxqn�R  
�          @���@�p�?&ff@�At(�@�Q�@�p���z�@
=A~=qC��q                                    Bxqn��  �          @���@�33��ff@(��A���C���@�33��
?���AZ�RC��)                                    Bxqo�  �          @�  @��\@[�A��HC���@��6ff@(�A�ffC�n                                    BxqoD  �          @��
@ָR����@UA�p�C��R@ָR�޸R@7�A�  C��q                                    Bxqo+�  �          @��R@�  �L��@e�A�Q�C�)@�  ��(�@H��A�{C���                                    Bxqo:�  
�          A ��@�=q�3�
@��HA�\)C��@�=q����@�RA�p�C�W
                                    BxqoI6  
Z          A (�@�\)��@��\BC���@�\)��G�?�Q�AC�C�t{                                    BxqoW�  �          A�@333��p�@���B{C��@333��G�?���A5��C���                                    Bxqof�  T          A��@z=q��Q�@L(�A�ffC�� @z=q��  =���?+�C�,�                                    Bxqou(  �          A=q@�p���(�@   A�\)C��{@�p���=q�z��|��C��                                    Bxqo��  T          A  @�
=��ff@#�
A�  C�{@�
=����G��AG�C�                                      Bxqo�t  
�          A�@�p���
=@�A^{C�Ǯ@�p���Q�333��Q�C�{                                    Bxqo�  "          A��@����Å?��HAT  C�ff@����˅�Tz�����C���                                    Bxqo��             A
�R@������@ffA`Q�C�C�@�����=q�&ff��G�C���                                    Bxqo�f  �          AQ�@�=q���\@\)An=qC�3@�=q��\)��ff�<��C�!H                                    Bxqo�  "          AG�@�\�33?xQ�@��HC��
@�\���,������C��                                    Bxqo۲  �          A�@�z�?\(�@��
C��@����5���p�C�@                                     Bxqo�X  �          A��@e��
=>���@�C�!H@e���
�Dz���(�C�H                                    Bxqo��  �          A�R@�p���
=>�G�@3�
C��@�p����7
=��  C��=                                    Bxqp�  
�          A(�@�33���H?c�
@�p�C��f@�33��\)�\)����C�R                                    BxqpJ  T          A33@��H������\�Q�C���@��H�ۅ�p  ��(�C�#�                                    Bxqp$�  �          A  @���أ׿�=q�BffC��@����(���Q���p�C��                                    Bxqp3�  �          A
�R@�Q���p���  ��C�^�@�Q��HQ���(��8Q�C���                                    BxqpB<  �          Az�@\����ff������z�C���@\���ָR��p���RC���                                    BxqpP�  "          A(�@-p���G��)����ffC��@-p��������$�
C���                                    Bxqp_�  T          A	G�@���ff��z��2�\C�]q@���ָR�����\C���                                    Bxqpn.  T          A
=@   ��
���v�RC�p�@   ��  ���H�{C��                                    Bxqp|�  �          A@�H���)������C�U�@�H��\)��Q��#{C��
                                    Bxqp�z  "          A=q?�(���R�(��~�HC��?�(�������
=� (�C�9�                                    Bxqp�   �          A�\?˅���B�\��z�C�.?˅���
�����C��                                    Bxqp��  
�          A�?�
=�\)�J=q��(�C��?�
=��33���\��  C�XR                                    Bxqp�l  �          A?��
�\)�O\)��  C�0�?��
��33���H��33C�˅                                    Bxqp�  
�          AQ�@����\<��
=�G�C��3@������]p���Q�C���                                    BxqpԸ  �          A
{@z�H��=q?��@�C���@z�H���H�	���hQ�C��                                    Bxqp�^  �          A	��@dz����R?�(�AffC�g�@dz������z��_
=C���                                    Bxqp�  �          Az�@QG����?k�@�ffC�^�@QG���  ����\)C��=                                    Bxqq �  �          A
=@3�
��p���\)��z�C�@3�
���`����33C��f                                    BxqqP  
�          A33@�ff��ff?aG�@�Q�C�#�@�ff���	���mC��                                     Bxqq�  T          A�@�=q���?��
AM�C�f@�=q��33�fff���C���                                    Bxqq,�  T          A=q@�����
=?��AV�HC��\@�����p��k��ϮC�#�                                    Bxqq;B  
Z          Ap�@��\�Ϯ?�A"ffC���@��\��Q쿬����C��3                                    BxqqI�  �          A  @����ff?�A;�C���@���ٙ������C���                                    BxqqX�  
�          Ap�@�{��  @��A�Q�C��=@�{��z����u�C�33                                    Bxqqg4  
�          A�@��\����@$z�A��C���@��\�������1�C��
                                    Bxqqu�  �          A�H@e��ڏ\@(��A�33C�ٚ@e���녾�33�\)C�\                                    Bxqq��  �          A�R@n�R���@{A|��C�0�@n�R��
=�B�\���
C���                                    Bxqq�&  �          Az�@w��ۅ@�RA�  C���@w����ÿ   �^{C���                                    Bxqq��  �          A�H@�=q���?xQ�@��HC��@�=q��\��
�b�HC��3                                    Bxqq�r  
�          A�@h����?��HA33C��q@h�����Ϳ�
=�S\)C�R                                    Bxqq�  �          A  @L����=q?@  @�33C�)@L����R�����z�C��
                                    Bxqq;  
(          A{@!G���
=>u?��C�@!G���z��>�R��
=C�e                                    Bxqq�d  "          A  @���(��L����33C���@������x����=qC���                                    Bxqq�
  T          A\)@�
=��z�@;�A�
=C�aH@�
=���?:�H@��
C�Z�                                    Bxqq��  
�          Az�@��\�>{@�B9�C�33@��\���H@�  A���C��q                                    BxqrV  �          A��@p�����@�Q�B\)C��@p���ۅ@�RA�z�C�`                                     Bxqr�  	�          A�@[����
@�  BC�'�@[�����@G�Aa��C��                                     Bxqr%�  T          A�@fff����@��B  C�\)@fff���H?��HA=�C�                                    Bxqr4H  �          A�?�\)��z�@���A�p�C���?�\)� Q�?�Q�A�
C���                                    BxqrB�  �          A��@4z�����@�=qA�33C�z�@4z���
=?���@�z�C��                                    BxqrQ�  �          A��@s33��ff@<(�A�\)C�Ǯ@s33���=L��>\C��R                                    Bxqr`:  �          A�@�  ��=q@J�HA��C��)@�  ��z�?O\)@��
C��\                                    Bxqrn�  �          Aff@1G������{�ffC��=@1G���(��[���G�C��H                                    Bxqr}�  �          A�
@h����  ?�ff@��C��@h����녿��R�[33C�9�                                    Bxqr�,  T          Ap�@_\)��G�>�z�?�p�C�n@_\)��G��,����p�C�/\                                    Bxqr��  
�          A��@O\)��=q�����1G�C��R@O\)����U����C��)                                    Bxqr�x  T          A��@j�H��z�?��AT(�C��q@j�H��녿����33C�W
                                    Bxqr�  T          A�
@�33��
=?
=@�(�C���@�33��z����uG�C��)                                    Bxqr��  
�          A33@�z��S33@l(�A�
=C���@�z���\)@
=Ax(�C��
                                    Bxqr�j  
�          A�H@ۅ>��
@��\B �@,(�@ۅ���@��A�C�E                                    Bxqr�  �          A�@�z�?�\)@�\)B�A4z�@�z��@��B{C�                                    Bxqr�  �          A�R@�\)?���@��HA�\A(��@�\)��33@�G�A�G�C��H                                    Bxqs\  �          A�R@�p�?�(�@�(�A뙚A9G�@�p���  @�(�A�33C��                                    Bxqs  T          A��@�p�?�z�@g�A�{An=q@�p�>�Q�@��\A��@6ff                                    Bxqs�  T          A�@�(�@?\)@�  B�A�ff@�(�?��\@��B"
=A�                                    Bxqs-N  �          AG�@�{?��@�
=BffA ��@�{��@��HBffC�s3                                    Bxqs;�  T          @�@����U@�RA�\)C��{@����|(�?E�@�ffC��q                                    BxqsJ�  �          @�G�@,����?��
@�z�C�K�@,�����ÿ��XQ�C��H                                    BxqsY@  �          A   ?��
����?Y��@��
C���?��
��Q������(�C�                                    Bxqsg�  �          A   @1G���
=>\@,��C�.@1G��ᙚ�!G���
=C��q                                    Bxqsv�  T          A��?�  ���>\)?xQ�C�\)?�  ���9�����HC���                                    Bxqs�2  
�          A�?��\��ff�\)�~{C��=?��\��(��c33���HC��                                    Bxqs��  "          A��@Q����
<#�
=�\)C�ff@Q������?\)���RC�
=                                    Bxqs�~  �          A@��(�>B�\?��C�E@���4z���z�C���                                    Bxqs�$  �          Aff@   �녽��Ϳ0��C��@   ��
=�K�����C��                                     Bxqs��  
�          A�H@����R?�33@���C���@��������H�Xz�C�&f                                    Bxqs�p  
Z          A�
@8Q���ff�L�;���C���@8Q����H�C33��p�C���                                    Bxqs�  T          A	G�@#33��H=���?(��C���@#33���@  ���C�B�                                    Bxqs�  
�          A�H@%��
=?333@�G�C�  @%�����
=��(�C�Z�                                    Bxqs�b  
�          A
=?�=q��R?\)@vffC�L�?�=q�����#33��C���                                    Bxqt	  
�          AG�@]p����?���A8(�C��@]p���z῕�=qC���                                    Bxqt�  
�          @�\)@�(���ff@dz�B\)C�u�@�(��z�@:=qA�C�Y�                                    Bxqt&T  �          A{@���@*�H@��B(�AΏ\@���?^�R@��B'  A�\                                    Bxqt4�  
�          A  @�  @p�@��
B
(�A���@�  ?��@��RB�R@���                                    BxqtC�  
�          A��@�p�@XQ�@x��A�p�A�z�@�p�?�Q�@��
B�Al                                      BxqtRF  
I          A\)@��H@���@L(�A�=qB'@��H@^{@��HB=qA��R                                    Bxqt`�  
�          A@�=q@��?���A\��B3�@�=q@�  @l(�A�Bff                                    Bxqto�  "          AG�@�(�>L��@��Bz�?�  @�(���p�@�  B {C��{                                    Bxqt~8  T          A�
@��ÿ�  @�(�B�RC���@����/\)@��A�\)C��                                     Bxqt��  
�          AG�@�  @��@�z�B�A�G�@�  >L��@��B0��?�Q�                                    Bxqt��  
�          A�@�(�@G�@�Q�B�HA��H@�(�>��@��B'p�@p�                                    Bxqt�*  �          Ap�@Ǯ?�@�ffBA�\)@Ǯ�u@�  B%�\C���                                    Bxqt��  T          A{@��ÿ   @��HB%��C��@�����@���BG�C�
=                                    Bxqt�v  T          @�ff@�{�p��@�{B&�
C��@�{����@\(�AѮC��R                                    Bxqt�  "          @�{@ҏ\?�{@eA�
=A[�@ҏ\>8Q�@{�A�Q�?���                                    Bxqt��  
Z          A  @�33@�p�>�ff@I��B��@�33@���@�
A��B��                                    Bxqt�h  
�          A   @�R@$z�@z�AqA���@�R?��@7
=A�\)AL��                                    Bxqu  T          A ��@�>��@+�A�\)@J=q@��
=@)��A�G�C��)                                    Bxqu�  �          @��@�R?Q�@Dz�A���@���@�R��33@J=qA�G�C���                                    BxquZ  	�          @�{@�p��#�
@k�A�RC��\@�p�����@Z�HA��
C��                                    Bxqu.   �          @���@�=q?��@j�HA�=qAS
=@�=q=���@~�RA��?fff                                    Bxqu<�  �          @�33@�(�@J=q@P  A�{A�z�@�(�?�G�@�p�B Aw\)                                    BxquKL  
�          @��R@ȣ�@W�@^�RA�=qA�=q@ȣ�?��@��RB	{A�                                      BxquY�  �          @��
@�
=@  @r�\A��
A�G�@�
=?:�H@�33B��@��                                    Bxquh�  �          @��H@���@2�\@}p�A�
=A�G�@���?�
=@�ffB�\A-G�                                    Bxquw>  
�          @��
@�@,��@J�HA��
A�(�@�?���@{�A�p�A:�H                                    Bxqu��            @��@�{@h��@\��A�
=A�  @�{@
=q@���B��A��                                    Bxqu��  T          A�@�\)�l��@!�A�  C��R@�\)���
?��@��
C��f                                    Bxqu�0  �          Az�@�(��mp�@O\)A��RC�` @�(���(�?ٙ�A=��C���                                    Bxqu��  T          AQ�@�(��i��@�  A��C��q@�(���@*=qA��C��)                                    Bxqu�|  T          A(�@����1�@��B�C�e@������
@j�HA�z�C���                                    Bxqu�"  T          A�@����ff@�ffBC�,�@���Vff@p  A��C�5�                                    Bxqu��  
�          A=q@�����@�=qBG�\C���@���N{@���B&p�C���                                    Bxqu�n  T          A�@�(����@Z�HA�p�C�9�@�(���Q�?�  AH  C�\)                                    Bxqu�  "          A33@�{��=q?�(�AQ�C�,�@�{��(����R� ��C�y�                                    Bxqv	�  
�          A\)@����׮@   AN�RC��@�����Q��\�Q�C�!H                                    Bxqv`  
�          A�@�ff��Q�?�
=ADz�C��\@�ff���(����{C�|)                                    Bxqv'  T          A=q@�{���?(�@vffC�e@�{��׿��R�I�C��                                     Bxqv5�  "          A�H@������@z�Aj�HC���@����  ��(��,��C��                                    BxqvDR  "          A�
@�����
=@{A^�HC��@�����G���ff�5�C���                                    BxqvR�  
Z          A  @�ff��(�@�AT��C���@�ff��p���\�K�C�%                                    Bxqva�  
Z          A  @��H��@�AK
=C�5�@��H���!G��{�C���                                    BxqvpD  T          Az�@�  ��\)@"�\A�ffC��\@�  ��
=>W
=?��C�xR                                    Bxqv~�  �          A��@�=q���?���@�Q�C���@�=q��\��{� (�C��                                    Bxqv��  "          A{@����\)?W
=@���C�O\@���陚���7�C��H                                    Bxqv�6  �          A@�  ��(�?Y��@�33C��R@�  ��R��\�3
=C�f                                    Bxqv��  �          A\)@�G����
?�@���C�Q�@�G���\��\)�z�C�b�                                    Bxqv��  �          AQ�@�(���G�?��HAG�C��\@�(����������HC���                                    Bxqv�(  "          AQ�@�����\>��@B�\C�k�@����Q�����iC��                                    Bxqv��  "          A
ff@C�
� �Ϳ0�����C�g�@C�
��=q�Z�H��z�C�W
                                    Bxqv�t  T          A
ff@
�H�33����  C�h�@
�H��  ������
=C�T{                                    Bxqv�  T          A�
@�G���?�Q�AF�\C��
@�G�����>�?L��C�H                                    Bxqw�  T          A(�@�=q��p�?��Ap�C�@ @�=q��zᾣ�
���HC���                                    Bxqwf  �          A�@����Ǯ?���@ٙ�C�u�@�����  ������C�o\                                    Bxqw   
�          A��@�ff��Q�?   @>�RC�l�@�ff��G���G��)�C��                                    Bxqw.�  �          AQ�@ʏ\��  �k���
=C��3@ʏ\�ȣ��!G��|z�C���                                    Bxqw=X  �          A
=@�p�����>8Q�?�
=C��3@�p��љ����a��C�n                                    BxqwK�  
�          A�\@����p�<#�
=�\)C��H@����Q���s�C�\)                                    BxqwZ�  	�          A�R@�  ��R��G��Q�C�Z�@�  �ȣ��w����C�C�                                    BxqwiJ  �          A�@����Q���F{C�@����������33C�7
                                    Bxqww�  
�          A@�  ���
����yG�C�h�@�  �Å��G���  C��                                    Bxqw��  T          A�@��\��Q�����tQ�C���@��\��Q���G����C��                                    Bxqw�<  
�          A�H@��\���0����{C��\@��\��\)�����HC�`                                     Bxqw��  T          Aff@����\)��H�u�C��R@����\)��G����C�g�                                    Bxqw��  T          AQ�@�
=��\)�33�l  C�p�@�
=������p����C��)                                    Bxqw�.  
�          A�@�����G��>�R���C���@����������\�p�C��                                    Bxqw��  "          A�
@�������XQ���33C���@��������=q��
C��                                    Bxqw�z  �          Az�@�����G��!��}�C�W
@�����=q��=q���C�H�                                    Bxqw�   
�          A��@��\��z��j=q��{C��@��\�J=q��  ��RC�=q                                    Bxqw��  �          A(�@�������
=�LQ�C�(�@����ə�������C�,�                                    Bxqx
l  �          A��@|(��{�L�Ϳ��
C��\@|(���33�8����
=C��{                                    Bxqx  "          A�\@�\)��
==�?B�\C���@�\)����!G���=qC�P�                                    Bxqx'�  �          A�@��
���?&ff@���C�#�@��
���   �Ip�C���                                    Bxqx6^  �          A(�@�\)���;���:�HC�s3@�\)����C33��z�C�u�                                    BxqxE  "          A�H@��p���=q����C��@���G��l(�����C���                                    BxqxS�  �          A�@��������*=qC���@����Q����
����C�g�                                    BxqxbP  
�          A33@u��
=?c�
@�Q�C���@u��=q��\�7�
C��                                    Bxqxp�  �          A�\@o\)��@Q�As\)C��)@o\)�zᾳ33��C�%                                    Bxqx�  �          A�
@�����R��Q��  C��@�������<(���33C��3                                    Bxqx�B  T          A�\@�p���\)�aG���33C��\@�p������QG���z�C��q                                    Bxqx��  �          A33@����녿��H�\)C��\@������z�H��C�~�                                    Bxqx��  "          A�@�{��33��{�=��C��
@�{���H������(�C���                                    Bxqx�4  �          A@�Q���������G�C�H@�Q��p  �ƸR�333C���                                    Bxqx��  �          A(�@�����
=��\)�p�C���@����;��ٙ��LG�C��                                    Bxqx׀  
�          AG�@����\)�����%�\C�q�@���5���33�X
=C�W
                                    Bxqx�&  T          A�@qG����������p�C��@qG����Ǯ�-\)C�aH                                    Bxqx��  �          A
=@b�\���ff�n{C��@b�\����33��G�C��f                                    Bxqyr  
�          A  @�=q�\)�׮�J=qC�3@�=q=#�
��33�Y�?�\                                    Bxqy  T          A	�@����W
=��{�'��C�+�@�����Q���\)�G��C�\                                    Bxqy �  
(          A�H@�G���녿�
=�)C�*=@�G���p��qG����C�,�                                    Bxqy/d  
�          A��@���ҏ\?�\)@�C��{@����33�����ffC���                                    Bxqy>
  
�          A�@���ڏ\@�\Ad  C�j=@����R�#�
�uC��=                                    BxqyL�  �          Ap�@����G�@Q�AR�HC��\@����=q�Ǯ���C�s3                                    Bxqy[V  �          A33@����(�?��HA33C�)@����ff������C���                                    Bxqyi�  T          A��@�\)��\)�(Q���ffC���@�\)��Q����
�\)C�                                    Bxqyx�  �          Az�@������
��(��E�C��@����Ӆ�����ݙ�C���                                    Bxqy�H  T          Aff@��
��ff�4z�����C��R@��
��p�������C�%                                    Bxqy��  
�          AG�@�Q���(���Q��;�
C�:�@�Q��ۅ���\��  C��                                    Bxqy��  �          A��@�ff���k�����C���@�ff��\)�W���C��q                                    Bxqy�:  �          A��@�z�����\)�ٙ�C���@�z���\�3�
��Q�C�n                                    Bxqy��  �          AG�@����=q@_\)A���C�7
@����=q?�@���C�Ǯ                                    BxqyІ  
�          Ap�@�G���
=@�{AݮC�=q@�G�� (�@�ADz�C�n                                    Bxqy�,  "          A�\@�z���G�@j=qA�\)C��H@�z����H?�{A��C�aH                                    Bxqy��  
�          A�\@�=q��z�@c�
A��C���@�=q�ָR?��A Q�C��R                                    Bxqy�x  "          A��@�Q���\)@o\)AɅC�5�@�Q����
?�AAC��                                    Bxqz  �          Aff@�33���H@n{A��HC��{@�33����@
=A\Q�C��                                    Bxqz�  "          Az�@�ff���R@qG�A˅C��\@�ff��p�@G�ApQ�C�,�                                    Bxqz(j  "          AQ�@ָR��(�?���A�C���@ָR������Q��ffC�8R                                    Bxqz7  �          A33@�{��(�������C���@�{���
�Vff��\)C�t{                                    BxqzE�  �          A33@�  ��
=�C�
���C�O\@�  ��  �����HC��f                                    BxqzT\  �          Ap�@�\)�����\�R{C���@�\)��33�c33���HC��                                    Bxqzc  �          A��@��H��Q�?�z�A+
=C���@��H����:�H��{C�|)                                    Bxqzq�  �          A�
@�
=��Q��
=�+�C�ٚ@�
=�����(Q�����C��=                                    Bxqz�N  �          A�R@��
��(������33C��f@��
��{�Vff��C��                                    Bxqz��  "          A��@�ff���ÿ�33�C33C�q@�ff���H��G���\)C��)                                    Bxqz��  �          A�@����  ��ff�6ffC�q@����Q��(�����RC��                                    Bxqz�@  T          A=q@�  ��z���)G�C�
=@�  �����u��z�C���                                    Bxqz��  
�          A=q@�
=��
=?�AQ�C�Y�@�
=��G�>.{?�{C��                                    BxqzɌ  "          A
=@ҏ\��{?!G�@�G�C��@ҏ\��33���H��33C�*=                                    Bxqz�2  
�          A��@�\)��(��(���=qC��H@�\)��
=��
�_�C��                                    Bxqz��  "          A��@����=q�.{���RC�� @�����H�=q���HC�R                                    Bxqz�~  �          A
=@�p����@��AyG�C��H@�p���p�>��?z�HC��f                                    Bxq{$  
�          A�@�(���\@
�HA[�C��q@�(����ͽ�\)��(�C�                                      Bxq{�  
�          A=q@�Q���ff=���?�RC���@�Q���(��	���Z�\C�5�                                    Bxq{!p  "          A(�@�����G�?J=q@�p�C�3@����ָR����G�C�=q                                    Bxq{0  �          A��@�=q��p���  �˅C�ff@�=q�����V�HC�B�                                    Bxq{>�  
�          A�\@�Q����H���U�C��H@�Q���z��=q�up�C��\                                    Bxq{Mb  �          A(�@�p��˅�z��fffC�.@�p���z��p��w
=C�C�                                    Bxq{\  �          A�\@����\)?���A	��C�l�@�����ff=�\)>�ffC��\                                    Bxq{j�  �          AQ�@�ff��Q�?��@]p�C�]q@�ff��(�����C���                                    Bxq{yT  "          A��@�Q��ٙ�<#�
=#�
C��q@�Q��Ϯ�G��M�C�ff                                    Bxq{��  
(          A\)@��H��33@c33A�G�C�g�@��H��
=@
�HA\(�C���                                    Bxq{��  
(          A�\@�
=��p�@��Ar{C�@�
=��z�?+�@�{C��                                    Bxq{�F  T          Az�@�p����H?���A9G�C���@�p��Ӆ��G��5C�
                                    Bxq{��  "          A�@������H?fff@�z�C��3@�����׿���33C��3                                    Bxq{  
Z          A�@�{��=q>�?Y��C��@�{����
=�[\)C�l�                                    Bxq{�8  �          A�@���(��"�\����C�xR@����z����
�Q�C��3                                    Bxq{��  T          A\)�.{��  ���\��\)C�]q�.{��z���p��933C�33                                    Bxq{�  "          A�?���\)�c�
����C��f?��ҏ\����!  C��                                    Bxq{�*  �          A
=@   �\)��\)��G�C�  @   �   �hQ���33C���                                    Bxq|�  �          A�@;��33�Ǯ�!G�C���@;���
=�6ff����C�
                                    Bxq|v  "          A=q@P���{��R����C���@P�����H�>{��{C���                                    Bxq|)  �          A=q@��\����@j=qAř�C�P�@��\��\)?�AB�RC�aH                                    Bxq|7�  �          A��@��\����@�=qA�33C�=q@��\��  @ffAz�RC��q                                    Bxq|Fh  �          A	p�@�z��z�@�ffBG
=C�U�@�z��hQ�@�(�B%��C���                                    Bxq|U  "          A
�R@�녿�\@�  Be  C�ٚ@���s33@�p�B@�C�P�                                    Bxq|c�  �          A	�@��
���@޸RB`{C��H@��
��@�Q�B6C�Ff                                    Bxq|rZ  
�          @�  @+��fff@�\)BY  C�O\@+���  @�p�Bz�C��                                     Bxq|�   
Z          @��R@����}p�@w�A��HC�<)@������@#�
A���C�<)                                    Bxq|��  T          @�\)@��
��G�@B�\A���C�@��
��
=?���ANffC��H                                    Bxq|�L  �          A
�H?��H��p��
�H��=qC���?��H��  ��=q�z�C��
                                    Bxq|��  �          A�þaG���z���
=�z�C�)�aG����
�����bQ�C��H                                    Bxq|��  
�          AQ�aG��ڏ\��
=�)33C�R�aG���Q���p��k�C��3                                    Bxq|�>  �          A�>.{��Q���z��$�
C��3>.{��ff��z��gz�C��q                                    Bxq|��  
Z          A��?@  ��33����	��C���?@  ������ff�K��C���                                    Bxq|�  T          A��?.{��Q������=qC��
?.{��Q��߮�IQ�C�`                                     Bxq|�0  
�          A
=>k���(���\)�Q�C��H>k���Q����T��C�.                                    Bxq}�  
�          A��  �Å��p��.(�C�4{��  ��p���p��np�C}B�                                    Bxq}|  �          A��,(����R��z��W��Cn��,(��\)�W
C[ٚ                                    Bxq}""  T          A�q���{����C�Ce�
�q�����z��n��CT�                                    Bxq}0�  
�          A���\��=q�����;
=Cw���\�aG���
=�uCm
=                                    Bxq}?n  �          A���Ǯ��z����R�-  C� �Ǯ��ff��R�k��Cy��                                    Bxq}N  T          A=q�{��p���ff�+ffCw^��{��  ��(��f{CnL�                                    Bxq}\�  
Z          A��
=��\)��33�Q�Cz�
=��z���R�X�Cs\                                    Bxq}k`  �          AG��Q���(���\)�*�Co���Q��n{��=q�_��Cd�)                                    Bxq}z  "          Ap���{��z����
����C��Ϳ�{��G��У��>z�C�g�                                    Bxq}��  "          A33>B�\��\�i������C���>B�\�ٙ���G��!�\C��\                                    Bxq}�R  �          A��>�
=����b�\��=qC�t{>�
=��\)��\)�33C��)                                    Bxq}��  T          A?+��33�S33��z�C�AH?+�����G��{C��f                                    Bxq}��  
Z          A��Ǯ��{���=��C��þǮ�c�
��R�~\)C���                                    Bxq}�D  �          Az��33�E���{�3Cqٚ��33�aG��   �RCP�                                    Bxq}��  	�          Az�   ��z��˅�K��C�:�   �J�H��G��C��f                                    Bxq}��  �          A?�
=���X�����C�]q?�
=��ff���\��RC��{                                    Bxq}�6  
�          A(�?�����33�����=qC��?����������/z�C�,�                                    Bxq}��  �          A�?�33������H��(�C�*=?�33�\��G��9=qC�u�                                    Bxq~�            A�
?��H�������
���C�l�?��H��=q��\�CQ�C��f                                    Bxq~(  �          AG�@p���R�;���G�C�B�@p��˅��(���
C���                                    Bxq~)�  �          A(�@�
=��(�?}p�@�ffC�\)@�
=��(����\���HC�]q                                    Bxq~8t  
�          A�\@��\���
��  �0��C�H�@��\�Ӆ�n{��Q�C���                                    Bxq~G  �          A(�@�\)���\������  C���@�\)��ff�S�
��ffC���                                    Bxq~U�  �          A\)@�Q����H�u��Q�C���@�Q�����p��bffC�u�                                    Bxq~df  "          A�
@�
=��p�?�G�A0��C��=@�
=�������%C�z�                                    Bxq~s  "          AQ�@�ff��
=?�p�AEG�C���@�ff��
=�aG�����C�Ff                                    Bxq~��  T          A��@�����{@ffAR=qC��@�����\)���
��G�C��{                                    Bxq~�X  
�          A��@�{��z�?�=qA6=qC�q�@�{��33���
� ��C�R                                    Bxq~��  
�          A�
@����  @��Ap(�C�U�@�����
>�  ?ǮC��R                                    Bxq~��  "          A
=@Tz��G�@.{A�z�C�'�@Tz��Q�>�
=@'
=C���                                    Bxq~�J  
�          Aff@�Q���R@aG�A�=qC�J=@�Q���z�?�A  C�,�                                    Bxq~��  �          A  @�
=��G�@C�
A��C��H@�
=���
?��@��C��                                    Bxq~ٖ  T          A�@��
���?��AC�C�@��
��  ����� ��C��                                    Bxq~�<  
Z          A\)@�(���33?���AB�HC���@�(�����>aG�?���C��q                                    Bxq~��  �          A��@������H@#�
A�C���@����љ�?O\)@�C��{                                    Bxq�  
�          AG�@��H�ƸR@5A��
C���@��H�׮?�ff@ϮC��q                                    Bxq.  "          A@�����z�?��A5�C��@�����p�>#�
?s33C��                                    Bxq"�  
�          A��@�Q��У�@H��A��HC��R@�Q���(�?��\@�(�C�w
                                    Bxq1z  
Z          Aff@ʏ\��p�@n{A�z�C��3@ʏ\��
=@�AW33C��)                                    Bxq@   T          A�@����(�@g
=A���C�  @����z�?��HAEG�C�O\                                    BxqN�  �          A�@\��Q�@XQ�A�z�C���@\��ff?��A$z�C�q                                    Bxq]l  �          A33@qG�?�A��Bz�\A�  @qG�����A��B��fC��{                                    Bxql  �          A�\@�����R@�  Bi�C�'�@����@�z�BYG�C���                                    Bxqz�  �          A{@��%�@�\)BA  C�O\@�����@���B�C�޸                                    Bxq�^  "          A��@���7
=@��B=��C�{@������@��B�C�                                    Bxq�  �          A\)@�
=�(�@�BB=qC�)@�
=�x��@�33B$Q�C�N                                    Bxq��  "          A�@�p��'
=@�p�B7=qC���@�p���\)@�  BffC��f                                    Bxq�P  T          A�\@����Z=q@��RB(\)C�q�@������
@��HB�HC��)                                    Bxq��  "          A@��R���@��HBz�C�� @��R��G�@_\)A��
C��                                    BxqҜ  T          Ap�@�=q��ff@�(�A���C��H@�=q����@R�\A��RC��q                                    Bxq�B  T          A@�\)��G�@�=qA�(�C���@�\)����@=p�A��RC�9�                                    Bxq��  
�          A33@����H@;�A��HC��{@����?���A(�C��                                    Bxq��  �          A�@���ڏ\?��
@��
C�Ф@���ۅ�O\)��
=C��H                                    Bxq�4  "          A�R@�
=���@\)A��C�Z�@�
=���?��RA33C��=                                    Bxq��  �          A�@������@A��HC�q�@����  ?���@�p�C��                                    Bxq�*�  "          A��@QG��p�?333@��C���@QG�������H�p�C��                                    Bxq�9&  "          A��?�(��z�?�  @ҏ\C��{?�(��  ���\��C���                                    Bxq�G�  "          A��@;��{?���@��C���@;��=q�������C��                                    Bxq�Vr  
�          A�
?��\)?n{@�(�C�#�?���\������C�.                                    Bxq�e  �          A��?G���
?p��@�(�C���?G��
=�����  C��{                                    Bxq�s�  
(          A�R?���	?���A"ffC�c�?�����=p���  C�Q�                                    Bxq��d  
�          A��@u��\)?���A�\C��{@u��녿.{��33C�s3                                    Bxq��
  �          A�@����{?�@p��C���@�����H��=q��C��H                                    Bxq���  "          A(�@{���(�@�RAn�HC��@{����R>k�?�G�C��H                                    Bxq��V  	�          A
=@�p���G�?�(�A8Q�C�H@�p��У�<�>8Q�C�}q                                    Bxq���  �          A	�@љ����@`  A��C���@љ���Q�@�Aw�C�K�                                    Bxq�ˢ  
�          A��@���  @I��A���C��@���p�?�Q�AR�\C��\                                    Bxq��H  T          A��@����33@8Q�A�(�C��R@������?��\A��C�Q�                                    Bxq���  
�          A��@�=q���R@\)A���C�˅@�=q���?k�@ƸRC���                                    Bxq���  "          A  @����p�@`  AÅC��
@����(�?���AIC�G�                                    Bxq�:  T          A  @�����p�@�G�A�{C�w
@�������@(Q�A�ffC��                                    Bxq��  
Z          Ap�@��?��\@�B<�A�=q@���L��@��HBDG�C��\                                    Bxq�#�  "          A33@XQ�@���@�\)Bp�Bi��@XQ�@�G�@ǮBG=qBH\)                                    Bxq�2,  "          A�H@s�
@�(�@�{B!��BU�@s�
@c�
@ҏ\BNffB,                                      Bxq�@�  �          A�
@��?��@�p�B5�A�
@����@ǮB8{C�H�                                    Bxq�Ox  
�          A33@�ff?У�@��
B$�
Ak�@�ff>�@�33B-33?���                                    Bxq�^  �          A�
@���?�
=@ʏ\B:z�A]@����u@�\)B@�\C��\                                    Bxq�l�  �          A@��R?���@�z�BB��A>=q@��R�
=q@�\)BF33C�H�                                    Bxq�{j  "          A(�@���?���@���B4��A���@���>���@�ffB@G�@@��                                    Bxq��  T          A\)@ə�@��@�B(�A�(�@ə�?z�H@�(�B*AG�                                    Bxq���  �          A��@�ff@(�@�{B$Q�A��@�ff?��@ə�B0��@���                                    Bxq��\  �          AQ�@�{?���@�ffB6�HA��@�{�(�@�Q�B9{C�,�                                    Bxq��  �          A�@�{>�
=@У�B>Q�@���@�{��G�@��B:33C�f                                    Bxq�Ĩ  
�          A	p�@��׿�p�@�
=B:�C���@����1G�@�ffB&��C�(�                                    Bxq��N  
�          Az�@�(��s33@�
=BG\)C��
@�(��$z�@���B5
=C�7
                                    Bxq���  �          A  @�녿˅@�ffB)(�C���@���?\)@�33BG�C��q                                    Bxq��  
Z          Aff@�{�=q@���BffC�\)@�{�g
=@�  A�33C��                                     Bxq��@  �          AG�@�p��(�@z=qA�Q�C�=q@�p��G
=@N{A�{C��=                                    Bxq��  �          A�@�
=����@QG�A��\C�t{@�
=��  @��At��C�O\                                    Bxq��  T          A��@�R��=q���
�
=C�Ǯ@�R���z��iG�C��                                    Bxq�+2  
Z          AG�@�ff�>�R@���B��C���@�ff��  @`  A�{C��                                    Bxq�9�  
�          Aff@��H�љ������	��C�=q@��H��Q�����]C��q                                    Bxq�H~  �          A�@�33��G��s33��Q�C��\@�33����1G���p�C���                                    Bxq�W$  �          Az�@�(���\�5���HC��{@�(�����"�\��{C�o\                                    Bxq�e�  T          AQ�@h����{>��R@
=C���@h����׿���1p�C���                                    Bxq�tp  �          A\)@N{��{?��@�{C�]q@N{��
=�c�
����C�T{                                    Bxq��  "          A\)@�\)��G�?��RA�RC�:�@�\)���
�z��{�C�3                                    Bxq���  
�          A
=@�{��p����^�RC�U�@�{�����{�Q��C��3                                    Bxq��b  �          A@�(������!G����C��@�(���{�}p���C��                                    Bxq��  
�          A�@�
=��(��P����G�C�=q@�
=��(���=q��\C�                                    Bxq���  "          A�@ڏ\�c�
�Q���ffC�:�@ڏ\�'���=q��\C�~�                                    Bxq��T  
�          A��@*=q���Ϳ����E�C�H�@*=q��{�p  ��=qC�!H                                    Bxq���  	�          A	��@C33��33����C�E@C33��Q���{��{C��                                    Bxq��  �          A\)@�(���z��AG�����C�:�@�(���p������  C�xR                                    Bxq��F  �          Aff@s�
��>��R@(�C��H@s�
���H��G��)�C�"�                                    Bxq��  �          A{@�����33?W
=@���C��{@������;���^�RC��3                                    Bxq��  �          A{@�����  @$z�A���C�ٚ@�����\)?�A33C��q                                    Bxq�$8  
�          A	G�@������B�\���C�j=@�����\)�����C��R                                    Bxq�2�  T          A	A�Ϳ��H��p��   C�]qA�Ϳ�ff�Y�����RC��                                     Bxq�A�  J          AA�þL��?��@z=qC�XRA�þ�33?�@]p�C��H                                    Bxq�P*  �          AQ�Aff����?���A
{C��Aff�!G�?�p�@��
C���                                    Bxq�^�  �          A�A33��=q?�
=AG�
C��A33�=p�?�ffA:�\C��3                                    Bxq�mv  
�          Az�A	�=��
?�A��?\)A	녾�=q?�33A
=C�R                                    Bxq�|  
(          A33A�\�(�?�
=AMC�� A�\�&ff?�=qA��C�+�                                    Bxq���  �          A
�R@�\)�J=q@
=Ab�RC�� @�\)�e?�G�A=qC���                                    Bxq��h  
�          A
�\@�=q�c33?��AK�C�n@�=q�z=q?s33@��HC�W
                                    Bxq��  �          A
�R@��
�^�R?n{@�p�C�\@��
�fff=#�
>uC���                                    Bxq���  "          A
�H@��Z�H?�ffA=qC��@��h��>�
=@4z�C�9�                                    Bxq��Z  �          A�
@θR�\)@�=qA��C�%@θR��{@L��A��C�N                                    Bxq��   
�          A�
@�(��I��@��HA�G�C�� @�(����@K�A���C�                                    Bxq��  	�          Aff@�ff���@�Q�B�
C�xR@�ff�P  @��\A��
C�'�                                    Bxq��L  �          A�R@�=q��@��
B	��C�
@�=q�Vff@z=qA�G�C�
=                                    Bxq���  �          Aff@Ǯ�?\)@���A�C�/\@Ǯ�}p�@Y��A��C��                                    Bxq��  �          A Q�@�  �S33@���A�
=C���@�  ��ff@G
=A�Q�C���                                    Bxq�>  "          @�ff@�G���(�@���B�RC��{@�G��\)@u�A�{C��                                    Bxq�+�  
�          @�ff@�녿\)@�(�B�HC�^�@�녿�  @��HB33C��
                                    Bxq�:�  "          AG�@����#�
@�p�BS��C�0�@����p�@�=qBC��C���                                    Bxq�I0  
�          AG�@��
���@�Q�Bd\)C�N@��
��@ϮBV�C�&f                                    Bxq�W�  
(          AG�@��>�@�33Bj(�@�Q�@�����R@�  Bd�HC��                                    Bxq�f|  �          A@�(��A�@�p�BIG�C�޸@�(���ff@�\)B#
=C�h�                                    Bxq�u"  �          A ��@�����p�@���B�C�xR@������@S33A£�C���                                    Bxq���  T          @�
=@�
=����@���BC�� @�
=��@A�A���C�>�                                    Bxq��n  
Z          @��R@R�\��  @8��A��C��f@R�\��Q�?�G�Az�C���                                    Bxq��  T          @�\)@�{����@UA�  C��@�{��{@�
A��
C��=                                    Bxq���  T          @�Q�@�����@�
A��C�U�@����\?���A�HC��                                    Bxq��`  �          @�z�@����=q>k�?�z�C��q@����ff���H�=qC�K�                                    Bxq��  T          @���@����z�=p�����C�� @��������p��h��C�w
                                    Bxq�۬  T          @�ff@Ǯ���R�z�H��z�C�xR@Ǯ���\�
=�x  C��                                     Bxq��R  "          @�{@�z����H��\)���C��
@�z����
��
=�C�
C�H�                                    Bxq���  	�          @�
=@   ��Q�>�\)@�C���@   ��33��G��5G�C��{                                    Bxq��  
�          @�{?������(���C��?�������{���C�E                                    Bxq�D  
�          A Q�@{��녿J=q��z�C��@{��z��'
=����C���                                    Bxq�$�  T          @��R@p��񙚿�(����C�,�@p�����B�\��\)C��H                                    Bxq�3�  "          @�ff@Q���׿���ffC���@Q���
=�E��  C�y�                                    Bxq�B6  
�          @�(�@7
=��
=��{�C���@7
=��\)�5���
C���                                    Bxq�P�  
�          @��H@4z���녿�{�?�C��@4z���ff�R�\��ffC��3                                    Bxq�_�  T          @�33@%���z�����W�C���@%���\)�`�����
C�ٚ                                    Bxq�n(  �          @�  ?�33�����Q���{C��\?�33�hQ������=qC�5�                                    Bxq�|�  
�          @�녿�33��
=�\����p�C��׿�33��{����(�C~�)                                    Bxq��t  �          @��R������׾\)��  C��
�����  ���o�
C���                                    Bxq��  
Z          A Q�s33����33�p��C�lͿs33�ڏ\�u���\C��                                    Bxq���  
�          @���:�H��
=�E��z�C���:�H�Ǯ��Q��p�C���                                    Bxq��f  "          @�녿���ff�8Q����C��3�����������Q�C���                                    Bxq��  
�          @����s33��(�����Q�C�XR�s33��33��Q���Q�C��=                                    Bxq�Բ  �          @�
==�Q���\�ff��
C�W
=�Q���33�u�����C�aH                                    Bxq��X  T          @�=q<�����
=�/\)C��<���G��K��ȣ�C�q                                    Bxq���  �          @�@HQ���(�?��HA3�
C��H@HQ����þL�Ϳ���C�`                                     Bxq� �  
�          @�  @C33�ٙ����ͿL��C��@C33��=q��G��ZffC�u�                                    Bxq�J  
(          @�@B�\��ff��(��VffC�0�@B�\��z��33��G�C��{                                    Bxq��  
�          @�ff@
=���H@�BC��@
=��=q@O\)A�G�C�9�                                    Bxq�,�  �          @�G�@ �����H@j�HA�  C�� @ �����@��A��\C��                                    Bxq�;<  
�          @陚?���Å@z�A�z�C�&f?���Ϯ?E�@�\)C���                                    Bxq�I�  
�          @�33@�\)��G�?���A.{C�{@�\)��ff��\)�z�C��{                                    Bxq�X�  T          @陚@����{?J=q@�  C�XR@����ff�.{���
C�P�                                    Bxq�g.  T          @�R@����\)@z�A��
C��f@�����H?J=q@�=qC�˅                                    Bxq�u�  �          @�@��R��(�@5�A�\)C���@��R��?�{AO�C��                                    Bxq��z  "          @�  @���Q�@5�A��HC��R@���w�?�Q�A}�C���                                    Bxq��   
�          @�33@������?�\)A�HC�P�@������
��\)�
=C��                                    Bxq���  "          @��
@ҏ\���@A���C��@ҏ\��=q@ ��A��\C�E                                    Bxq��l  T          @�@�{�S33?�Q�A~ffC�y�@�{�j�H?���A�
C�#�                                    Bxq��  �          @��H@u���\=�?xQ�C��@u����ff�,Q�C��                                    Bxq�͸  T          @��H@W
=���;��H��z�C��3@W
=��33��z�����C�xR                                    Bxq��^  
�          @���@p  ����u���RC���@p  ����{�[
=C�J=                                    Bxq��  
(          @�@�33��{������C���@�33�����  ����C�K�                                    Bxq���  T          @���@�Q��z�H>k�?�
=C��H@�Q��w
=�:�H���C�
                                    Bxq�P  T          @�Q�@��H�Y�������S33C�l�@��H�L�Ϳ�(�� ��C�&f                                    Bxq��  T          @��@��H�_\)>���@�C��@��H�\�Ϳ\)���\C�7
                                    Bxq�%�  
�          @��@�{���\?!G�@��C��H@�{��33��ff�j=qC��\                                    Bxq�4B  T          @�G�@�33��{<�>�\)C�� @�33��=q�}p����C��                                    Bxq�B�  "          @��@������>�p�@Dz�C��R@����Q�.{���
C�
                                    Bxq�Q�  T          @���@1G��ƸR>���@UC�f@1G����
�����z�C�.                                    Bxq�`4  �          @�G�@�ff���
?333@��C���@�ff����\�H��C�~�                                    Bxq�n�  
�          @�@������þu���HC��{@������\��33�7�C�33                                    Bxq�}�  �          @�33@�(��N�R�p�����C�!H@�(��9���ٙ��a�C�Z�                                    Bxq��&  
�          @��@�\)�'
=��Q��a�C��@�\)�������RC���                                    Bxq���  T          @�\)@ʏ\�8Q����\)C�Ǯ@ʏ\�1녿:�H���C�#�                                    Bxq��r  T          @���@Å�Dz�>\@L��C���@Å�Dz�\�J=qC���                                    Bxq��  
�          @��@mp���\)@!�A���C�
=@mp���?�\)A��C�H                                    Bxq�ƾ  	�          @�ff@Vff����@
�HA�{C��\@Vff��  ?+�@�z�C�)                                    Bxq��d  
�          @�
=@S�
����@��A�33C���@S�
��=q?p��@�
=C��\                                    Bxq��
  
�          @�
=@\�����H@(�A�{C�K�@\����Q�?p��@�\)C�q�                                    Bxq��  
�          @�@Mp����
@/\)A��C�U�@Mp����
?�(�A�C�c�                                    Bxq�V  T          @�@\�����H@:=qA��C�ٚ@\����(�?��HA;\)C���                                    Bxq��  
	          @�R@s�
��33?��RAAG�C���@s�
��G�<��
>.{C�"�                                    Bxq��  s          @�@��\���
?�=qA\)C���@��\��\)�W
=��33C�\)                                    Bxq�-H  �          @�=q@�z��(Q�?���A0z�C��@�z��8Q�?(��@��C��q                                    Bxq�;�  
�          @�@��
��p�?O\)@�G�C��@��
��
=����S�
C��                                    Bxq�J�  �          @�R@����\)?�A��C���@��������
�+�C�Y�                                    Bxq�Y:  �          @��
@�����Q�>�  @Q�C���@��������ff�p�C���                                    Bxq�g�  T          @�z�@QG�����?
=q@�ffC��q@QG����H�z�H� ��C�R                                    Bxq�v�  
�          @�@�\���;�ff�i��C�}q@�\���H��
��(�C���                                    Bxq��,  �          @�?��H��G��+���z�C��?��H��p���
��C�}q                                    Bxq���  T          @��H?O\)��  �\�C33C�L�?O\)��ff����
=C�s3                                    Bxq��x  �          @�33?�\)�ָR����(�C��=?�\)�Ǯ�*�H��(�C�Ff                                    Bxq��  
Z          @��@�����?B�\@��C��@����z�c�
��{C��3                                    Bxq���  
(          @�@�
�׮>.{?���C�w
@�
�ҏ\���R�A�C��\                                    Bxq��j  �          @��
@$z���녾W
=��C��
@$z���녿��mC��                                    Bxq��  �          @�\@��љ�?˅AO�
C��@��׮��\)��C���                                    Bxq��  T          @��H@%����?�p�A#�C�/\@%��zᾳ33�<(�C��                                    Bxq��\  	�          @��
@�����?z�@�C�ff@����(��Tz��ָRC�y�                                    Bxq�	  �          @�z�@J=q�ȣ�?�(�Ap�C�e@J=q��(���Q��8��C�33                                    Bxq��  �          @�p�@HQ���{?��AU�C�k�@HQ����=�\)?�C��                                    Bxq�&N  T          @���@N�R��=q?J=q@�(�C���@N�R��=q�G����C���                                    Bxq�4�  	�          @��@|����ff?�  Ab�\C�Y�@|�����R>�z�@ffC��                                    Bxq�C�  	          @�z�@S�
���@�A�  C���@S�
�ə�?L��@�ffC�ٚ                                    Bxq�R@  T          @���@0����@(�A�33C��@0������?��@�=qC�t{                                    Bxq�`�  �          @�ff?�\)��
=?�p�A^�HC��H?�\)��{<��
>.{C��                                    Bxq�o�  �          @��@   ����?���Aqp�C��@   ���>aG�?�  C�C�                                    Bxq�~2  �          @�{?��θR@�A�Q�C�E?����H?(��@�=qC��
                                    Bxq���  �          @�{@(�����@,(�A�(�C��
@(���  ?��A�C��R                                    Bxq��~  �          @�ff@33���H@@��AƏ\C�Z�@33��z�?�z�A5�C��                                    Bxq��$  �          @�R@Q��˅?�Q�A~�RC�@ @Q���z�>��R@�RC�ٚ                                    Bxq���  �          @�@$z���=u>�C��H@$z��Ϯ�����N{C��=                                    Bxq��p  �          @�@��
���H@eA�C��{@��
���H@A�\)C���                                    Bxq��  �          @�{@������@C33A�p�C�W
@������?ٙ�A[�C��                                    Bxq��  �          @�R@�\)����?���Apz�C���@�\)��=q?��@�p�C��                                    Bxq��b  �          @�{@�z���(�?�33A�C��3@�z���  ������RC�z�                                    Bxq�  �          @�
=@|(����H?���A:ffC�  @|(���Q�#�
����C���                                    Bxq��  T          @�
=@tz���\)?��\A33C�L�@tz��������H�|��C�'�                                    Bxq�T  �          @�R@�����=q?��@�z�C�9�@�����녿����G�C�>�                                    Bxq�-�  �          @�@�z����>�@w�C�� @�z����\�.{��\)C��3                                    Bxq�<�  �          @��
@������<#�
=�G�C��)@�����
=�����
=C�,�                                    Bxq�KF  �          @��H@�{�\(�=�Q�?5C�z�@�{�Vff�E���  C��=                                    Bxq�Y�  �          @�\@�\)�mp�>�  @G�C�{@�\)�j=q�.{��=qC�E                                    Bxq�h�  �          @��H@��R��z�?z�H@��RC�AH@��R��
=��{�.�RC��                                    Bxq�w8  �          @��@�����@�A��C���@�����G�?L��@��HC�˅                                    Bxq���  �          @�@����@%�A�G�C�.@���33?��
A'�C�Ф                                    Bxq���  �          @�\@�����
=��Q�:�HC��
@������׿�G��G�
C��                                    Bxq��*  �          @�G�@\(������33�G�C�q@\(����%���C�/\                                    Bxq���  
�          @�G�@c33��p�?��Ao�
C��@c33��{>��
@'�C�q�                                    Bxq��v  �          @��H@�R��p�>�Q�@:�HC�=q@�R�љ������-�C�g�                                    Bxq��  �          @ᙚ@����H�#�
��Q�C�  @����
��Q��_33C�s3                                    Bxq���  �          @�
=@4z��ə�>aG�?���C��@4z����Ϳ�\)�6�RC�N                                    Bxq��h  �          @��@�����
=?��HAD  C��@�����p�=�G�?c�
C�q                                    Bxq��  �          @��@���@�Q�?˅A\��B$33@���@n�R@'
=A�{B{                                    Bxq�	�  �          @ٙ�@�\)@-p�?�Q�Ag33A��@�\)@��@z�A�  A��                                    Bxq�Z  �          @ٙ�@��@&ff?��A=A��
@��@
�H@   A�A���                                    Bxq�'   T          @�=q@�\)��ff@{A���C�<)@�\)�У�@Q�A�Q�C��f                                    Bxq�5�  �          @��
@��Ϳ
=@�A�  C�b�@��Ϳ�z�@	��A�{C��R                                    Bxq�DL  �          @ڏ\@ƸR�W
=@�A�p�C��@ƸR�L��@  A��C�S3                                    Bxq�R�  T          @��H@�\)@.{>��H@��A�ff@�\)@   ?�z�Az�A�33                                    Bxq�a�  T          @��@�z�@W�>��?��A�(�@�z�@N{?�  A�A�\)                                    Bxq�p>  �          @ۅ@�33@���?E�@�\)B#33@�33@��?���A}�B=q                                    Bxq�~�  T          @ۅ@�z�?Ǯ�����XQ�AS�
@�z�?�{<#�
=L��AZ�\                                    Bxq���  T          @޸R@��
?Q녾���y��@��@��
?h�þ�  �33@�=q                                    Bxq��0  �          @�{@׮?�p��\)����AF{@׮?��H>u@G�AD(�                                    Bxq���  �          @��@�(�?���Q��A�Aw�@�(�?�\)=�G�?p��A{�                                    Bxq��|  �          @�(�@ָR?���    ���
AB�\@ָR?�33>�Q�@C33A<z�                                    Bxq��"  T          @���@ָR?}p�>u?��RA�\@ָR?fff>�@���@�z�                                    Bxq���  �          @�z�@��?���!G���G�@�  @��?=p����vff@�                                    Bxq��n  �          @ۅ@��
?У׿z����A]p�@��
?�p��\)���HAi                                    Bxq��  �          @��H@�p�?�{�#�
����A8��@�p�?�{>B�\?˅A8(�                                    Bxq��  �          @��H@�
=?�z�>�\)@ffA��@�
=?�ff?�@�Q�A{                                    Bxq�`  �          @��H@ָR?&ff?�  A�H@��@ָR>\?���A�@Mp�                                    Bxq�   �          @�z�@�p���=q?�z�A_\)C���@�p��0��?��AN=qC�\                                    Bxq�.�  �          @��H@�ff>�(�?�(�A$Q�@j=q@�ff=���?��A-?aG�                                    Bxq�=R  �          @أ�@ҏ\��33?h��@�
=C�\@ҏ\���?z�@�ffC�>�                                    Bxq�K�  
�          @�z�@�ff��\?���A0��C�<)@�ff��
?@  @���C�'�                                    Bxq�Z�  �          @���@�{�p�?uA Q�C��@�{���>�Q�@?\)C��
                                    Bxq�iD  �          @�(�@�
=��\)?�
=AFffC�� @�
=��?fff@���C�\)                                    Bxq�w�  �          @�@�p��w
=?8Q�@�
=C���@�p��z=q�����\)C��3                                    Bxq���  �          @�{@��
��
=>�ff@r�\C��@��
��ff�&ff��ffC�H                                    Bxq��6  �          @�  @�ff����?8Q�@�ffC�s3@�ff�\>�{@H��C���                                    Bxq���  �          @���@�Q�@<(�?�p�A�{A�(�@�Q�@�@*�HA�  A��                                    Bxq���  �          @���@��
?�G�?�33A���A�@��
>�@A���@��\                                    Bxq��(  �          @�  @ҏ\�u?�\)A��C���@ҏ\���H?Tz�@��HC��                                    Bxq���  �          @�
=@ƸR?J=q@p�A�p�@�@ƸR>B�\@A���?�                                      Bxq��t  �          @�(�@�G��k�@E�A�C���@�G���Q�@.�RA�Q�C�4{                                    Bxq��  �          @�(�@�(���Q�@^{A���C�%@�(���  @P  A�p�C���                                    Bxq���  T          @�@�{?�@�ffBE\)@ָR@�{�Y��@��BC(�C���                                    Bxq�
f  
�          @�z�@��@R�\@j�HB  B�R@��@{@���B!�RA˅                                    Bxq�  �          @��
@�\)@��@��
B33A�Q�@�\)?��@��B3�AG
=                                    Bxq�'�  �          @�@�G�?�=q@�33B2�
A��@�G�?   @��BA�
@ə�                                    Bxq�6X  �          @߮@�  ?��@��B'�AM�@�  ����@��B-�HC�ff                                    Bxq�D�  �          @߮@�
=>L��@�33B<�@Q�@�
=���@�
=B6��C�0�                                    Bxq�S�  �          @��@~{����@�  BP��C�5�@~{�<��@�z�B2�C���                                    Bxq�bJ  �          @�
=@|�Ϳ��R@�{BP�
C���@|���7
=@��B3��C�                                    Bxq�p�  �          @�\)@z=q�2�\@�z�B6(�C�=q@z=q�~�R@\)BQ�C�@                                     Bxq��  �          @�{@;��u�@���B3G�C��3@;����R@g�B �\C�K�                                    Bxq��<  �          @�p�@��\�]p�@`��A�\C�7
@��\��Q�@�RA�=qC�N                                    Bxq���  �          @�z�@�Q����@!G�A�p�C��@�Q�����?�z�A�RC���                                    Bxq���  �          @��H@��R���R?c�
@�=qC�� @��R���þǮ�Tz�C�K�                                    Bxq��.  �          @�@����\)�8Q�����C�
=@�����
������C�*=                                    Bxq���  �          @�ff@�G�����
=�>ffC�Q�@�G�����3�
���C�޸                                    Bxq��z  �          @��@������R��p����HC��)@��������J=q��(�C�C�                                    Bxq��   �          @�=q@Ǯ��?s33Az�C�o\@Ǯ�
=>��@a�C���                                    Bxq���  �          @�33@�p�����@+�A�z�C�z�@�p���\)@��A��\C��\                                    Bxq�l  �          @ۅ@�z��@��A��C���@�z῔z�@{A���C��
                                    Bxq�  �          @�33@�녿Tz�?ٙ�Ag33C�ff@�녿�  ?�A@Q�C��R                                    Bxq� �  �          @ۅ@��
�\?Q�@���C��@��
��Q�>Ǯ@N�RC�ٚ                                    Bxq�/^  T          @��H@�
=���>8Q�?\C��@�
=����.{��
=C���                                    Bxq�>  �          @�@�  �\=��
?&ffC���@�  ���R�����/\)C�Ǯ                                    Bxq�L�  �          @�@�(��\)>��
@(��C��@�(���R>#�
?�ffC�h�                                    Bxq�[P  �          @���@�녿B�\�
=q����C�Ф@�녿
=�:�H��=qC���                                    Bxq�i�  T          @�(�@�G�����(��#�C���@�G����
��\)�Z�\C�y�                                    Bxq�x�  �          @�z�@�ff��  ��
=�`��C��\@�ff��=q�O\)��  C�`                                     Bxq��B  �          @�@�
=��\)�8Q쿹��C�8R@�
=��G��(�����C���                                    Bxq���  �          @�
=@���>u@ ��C���@���=�Q�?B�\C��H                                    Bxq���  �          @�ff@���G��#�
���RC��@���Q콏\)�
=qC��)                                    Bxq��4  �          @�p�@�z�#�
>Ǯ@N�RC��3@�z���>�Q�@AG�C�e                                    Bxq���  �          @�p�@ۅ�G����
��C���@ۅ�=p��k����C��                                    Bxq�Ѐ  �          @޸R@�z�u=�G�?aG�C��@�z�s33�#�
���C��                                    Bxq��&  �          @�  @�33��z�=�\)?
=qC�1�@�33��\)���
�'
=C�W
                                    Bxq���  �          @�@�{���R>.{?���C���@�{��p��u����C���                                    Bxq��r  �          @ᙚ@��Ϳ���?�@���C��R@��Ϳ�z�>��?�
=C�9�                                    Bxq�  �          @�
=@��Ϳ=p�?.{@�(�C��3@��Ϳfff>�@p��C�L�                                    Bxq��  �          @�{@��
�Ǯ?O\)@�ffC�aH@��
���?+�@��C��f                                    Bxq�(d  
�          @�{@��H<��
?��A�>L��@��H��\)?��AQ�C��R                                    Bxq�7
  �          @�{@�=�\)?�Av�\?\)@���G�?��Ao\)C�)                                    Bxq�E�  �          @�ff@�Q�#�
@��A��
C��R@�Q�0��@33A���C��
                                    Bxq�TV  �          @��@˅�\)?�
=A�33C�` @˅�+�?���A~=qC��                                    Bxq�b�  �          @�Q�@߮��  =��
?&ffC��q@߮���<#�
=�\)C��                                    Bxq�q�  �          @ᙚ@�Q�>�33?#�
@�  @7
=@�Q�>#�
?8Q�@��\?�ff                                    Bxq��H  T          @���@�\)?.{>�(�@^�R@���@�\)?�?��@��
@��                                    Bxq���  �          @߮@���>�p�?c�
@�\@C33@���=�G�?u@�(�?c�
                                    Bxq���  �          @�\)@�{<#�
@ ��A�  >�@�{�.{@=qA�\)C��q                                    Bxq��:  �          @�ff@�
==�\)@ ��A��
?��@�
=�!G�@�A�=qC�5�                                    Bxq���  �          @�  @ָR��z�?�33A|z�C���@ָR�O\)?޸RAg33C��\                                    Bxq�Ɇ  T          @�Q�@�ff��{?L��@��C��
@�ff���?+�@�Q�C���                                    Bxq��,  T          @�  @�(���G�?�
=A�C�+�@�(��@  ?�G�A�C��                                    Bxq���  �          @�Q�@ٙ����?G�@�C��q@ٙ����H>Ǯ@L(�C��                                    Bxq��x  �          @߮@�z��=q�xQ�� ��C�Ff@�z῾�R�����@Q�C���                                    Bxq�  �          @߮@�(���?��
AQ�C�t{@�(����?s33@��
C�G�                                    Bxq��  �          @�\)@��;���?^�R@��C��q@��Ϳ�?@  @�p�C���                                    Bxq�!j  �          @��@ۅ�   >��H@��HC��@ۅ��R>���@0  C�g�                                    Bxq�0  �          @�z�@��<��
?uA Q�>8Q�@�녾�  ?n{@��C��3                                    Bxq�>�  �          @���@���>#�
?�G�A((�?�{@��þL��?�  A'\)C�'�                                    Bxq�M\  �          @���@�{>�?���AV�H@��H@�{<#�
?�A`z�=�\)                                    Bxq�\  �          @�p�@�33>u?��HA��@�@�33���
?�Q�A�
=C���                                    Bxq�j�  �          @�ff@�Q�>���?��
AL  @!�@�Q���?�ffAO
=C�aH                                    Bxq�yN  �          @��@ָR?�?}p�A��A=q@ָR?W
=?��A,Q�@�(�                                    Bxq���  �          @��
@Ӆ?�\?�@�z�Ao
=@Ӆ?��?�  A�RAR{                                    Bxq���  �          @��H@أ�?8Q�>�
=@a�@�=q@أ�?�?(�@��@�=q                                    Bxq��@  �          @ۅ@ٙ��u�����C���@ٙ����
�z���z�C���                                    Bxq���  �          @�(�@ڏ\�
=��
=�`��C���@ڏ\��ff������C��                                    Bxq�  �          @�(�@�G��s33�����C��@�G��B�\�E���{C�Ф                                    Bxq��2  �          @�z�@��H���Ϳ\)���C�Q�@��H�k��&ff��C�                                    Bxq���  �          @��
@�  ����>�z�@�HC��R@�  ��p���\)���C��
                                    Bxq��~  �          @�  @ٙ���z�?B�\@�
=C�%@ٙ�����>��
@(Q�C���                                    Bxq��$  �          @�\)@�녿���?333@�Q�C�� @�녿�(�>�z�@Q�C��=                                    Bxq��  �          @߮@�33��Q�?ǮANffC���@�33�z�?�G�A��C�P�                                    Bxq�p  �          @޸R@�\)��ff@ ��A���C���@�\)���?��Az�HC�C�                                    Bxq�)  �          @�p�@�{��z�@=p�A�G�C��@�{�,(�@�A��\C���                                    Bxq�7�  �          @�
=@�  �$z�@(��A���C��3@�  �N�R?��
Aqp�C�\)                                    Bxq�Fb  �          @�{@����@�A��C�W
@���9��?�33A<  C�<)                                    Bxq�U  T          @�(�@���33?�{A733C��@���
=?5@�p�C���                                    Bxq�c�  �          @���@�z��
=?�
=Aa��C���@�z��?��A\)C��{                                    Bxq�rT  �          @��@��(�?���AR�HC�g�@����?�=qA1��C���                                    Bxq���  �          @ۅ@ָR�(�?��
A+�C�k�@ָR�s33?��A(�C��
                                    Bxq���  �          @أ�@�ff�8Q�?Tz�@ᙚC�=q@�ff���?=p�@�G�C�=q                                    Bxq��F  �          @�(�@У׾#�
?���AC�J=@У׾��H?��A(�C��q                                    Bxq���  �          @�z�@�  �\?�p�A*�HC�XR@�  �8Q�?��A�
C��
                                    Bxq���  �          @�(�@�G��L��?n{Ap�C�!H@�G���?Tz�@�
=C���                                    Bxq��8  �          @�  @θR�8Q�?333@ƸRC�7
@θR�\?�R@�
=C�U�                                    Bxq���  �          @�{@�(����
?J=q@��C��@�(����R?:�H@���C��q                                    Bxq��  �          @�{@ə��O\)?}p�AG�C�Z�@ə����?333@�G�C�7
                                    Bxq��*  �          @�(�@�  ��
=?�@���C��q@�  ���>B�\?�z�C�aH                                    Bxq��  �          @���@�녿5?B�\@�G�C��@�녿fff?�\@�C�R                                    Bxq�v  �          @�33@�  �@  ?:�H@�z�C��{@�  �p��>�@���C���                                    Bxq�"  �          @�  @�(��0��?xQ�A
{C��@�(��s33?:�H@�p�C���                                    Bxq�0�  �          @��H@θR�8Q�?h��@�C��{@θR�u?&ff@�{C�˅                                    Bxq�?h  �          @�@�(��.{=u?�\C��@�(��(�þ����  C�#�                                    Bxq�N  �          @�(�@�=q��    �#�
C���@�=q�
=q�.{�\C��H                                    Bxq�\�  �          @�33@Ϯ�h��>k�@�\C��@Ϯ�n{�L�;�
=C���                                    Bxq�kZ  �          @љ�@�ff��ff>\@W�C�aH@�ff��\)=L��>�C�
                                    Bxq�z   �          @�=q@�zῼ(�>\)?��RC��f@�zῸQ쾞�R�*�HC���                                    Bxq���  �          @�=q@��Ϳ�33�����%�C��\@��Ϳ�  �333���C�xR                                    Bxq��L  �          @Ӆ@˅��>��?��\C�@˅��\����aG�C�8R                                    Bxq���  �          @�\)@�ff��=q������C�
@�ff�ٙ��+�����C���                                    Bxq���  �          @أ�@љ����
?�R@�  C�o\@љ����>�?���C���                                    Bxq��>  �          @��H@��
����\)�\)C���@��
��\�L���׮C��                                    Bxq���  
�          @���@����G��Ǯ�U�C�@����33�����C���                                    Bxq���  �          @أ�@�
=�c�
��p��M�C�u�@�
=�<(��   ��G�C��)                                    Bxq��0  �          @�(�@���j=q>u@�
C�#�@���c�
�c�
���C���                                    Bxq���  �          @�G�@����/\)?�
=A&ffC�^�@����=p�>�z�@ ��C�u�                                    Bxq�|  �          @�G�@�=q�.�R?�p�A.{C�p�@�=q�>{>�{@?\)C�u�                                    Bxq�"  �          @�G�@����.�R?�AH��C�T{@����A�?�\@���C�                                      Bxq�)�  �          @Ϯ@�
=���@	��A���C���@�
=��
=?У�Aip�C��                                    Bxq�8n  �          @Ϯ@�p���(�?��A\Q�C�l�@�p��У�?��A\)C���                                    Bxq�G  �          @�Q�@��ÿ��?L��@��C�Ǯ@��ÿǮ>���@;�C��                                    Bxq�U�  �          @���@Ǯ��>�@�=qC�@Ǯ��33�����C���                                    Bxq�d`  �          @θR@��\�(�@0  A��C���@��\�J�H?�{A��C���                                    Bxq�s  �          @��H@���<��@(�A�{C��R@���c�
?�z�AEC�5�                                    Bxq���  �          @��@��R�J=q@��A�p�C�g�@��R�o\)?�ffA6�\C�*=                                    Bxq��R  
�          @�p�@�
=�p�@ffA��C�Z�@�
=�Dz�?�(�ALQ�C��                                    Bxq���  �          @�(�@�  �9��@
�HA��C��@�  �[�?�33A!p�C��                                    Bxq���  �          @�
=@�G��R�\@
=A�33C�p�@�G��w
=?�(�A-p�C�G�                                    Bxq��D  �          @�z�@��
�,��@+�AƏ\C�{@��
�Y��?ٙ�Aw33C�/\                                    Bxq���  �          @�
=@���-p�@Q�A�Q�C�� @���Tz�?�AK�
C��                                    Bxq�ِ  �          @�{@��
=?�p�A{33C��3@��2�\?fffA��C��                                    Bxq��6  �          @�ff@�G��'�?fffA ��C��3@�G��1G�<�>�\)C�7
                                    Bxq���  �          @��H@�=q��{?�Ak�
C�u�@�=q��\?�  A  C���                                    Bxq��  �          @Ӆ@�33�/\)@ffA��HC���@�33�P��?���A=qC��                                    Bxq�(  �          @�G�@��4z�@333A�p�C���@��c�
?�\A{�C��H                                    Bxq�"�  �          @�z�@��R�Vff?�\)Ac
=C�8R@��R�l(�>�@�(�C���                                    Bxq�1t  �          @�G�@�(��e?Tz�@�=qC�&f@�(��j�H��33�Dz�C��q                                    Bxq�@  �          @Ϯ@���]p�?0��@�z�C��@���`  ��G��|(�C���                                    Bxq�N�  �          @�\)@�z��Tz�?��HA-p�C�+�@�z��b�\=�Q�?Q�C�]q                                    Bxq�]f  �          @θR@�p��aG�?��RAV=qC���@�p��s�
>�z�@%C���                                    Bxq�l  �          @�ff@��H�`  ?޸RA{�C��)@��H�w�?�@�\)C�]q                                    Bxq�z�  �          @�\)@����z=q@33A�=qC���@�����{?n{AG�C��                                    Bxq��X  �          @��H@�����z�@'
=A�C��3@����
�H@G�A�C��H                                    Bxq���  �          @��@�
=?}p�@;�A�Q�A$  @�
=    @EA��
C���                                    Bxq���  �          @�z�@�=q?=p�@Z=qB@�z�@�=q����@^{B�C�ٚ                                    Bxq��J  �          @�@�z�?�=q@.�RA�p�AS�
@�z�>\@@��A�  @u                                    Bxq���  �          @���@�?c�
@K�A�(�AQ�@��B�\@R�\A��C��                                    Bxq�Җ  �          @�\)@�
=?@  @
=A��@��@�
=�L��@�RA�ffC��q                                    Bxq��<  �          @�{@��?G�@N�RA��A�@�����R@S�
A��\C�ff                                    Bxq���  �          @�
=@�\)���R@I��A�(�C�u�@�\)���@8Q�A���C��=                                    Bxq���  �          @�G�@�\)�#33@AG�A�ffC���@�\)�Y��@�A���C��                                    Bxq�.  �          @ٙ�@�Q��\��@?\)A��C��R@�Q����?ٙ�Ah(�C��                                    Bxq��  �          @���@������\@(�A�ffC�k�@�����z�?c�
@��\C��                                    Bxq�*z  �          @�\)@~{��\)>Ǯ@\(�C���@~{��=q����;�C��                                    Bxq�9   �          @׮@����
@!�A���C��@����
>��@���C�/\                                    Bxq�G�  �          @��@B�\��p�@7�A�  C��H@B�\���?�
=A-G�C��                                    Bxq�Vl  �          @ָR@��R��{@�=qB-G�C�\)@��R�#33@|��Bz�C���                                    Bxq�e  �          @�{@����
=@\)B
=C��f@����b�\@?\)A�G�C��                                    Bxq�s�  �          @׮@i����
=@E�Aڏ\C�ٚ@i����
=?���A9�C��R                                    Bxq��^  �          @�=q@H����G�@��A�=qC�@H����\)>���@#33C�ٚ                                    Bxq��  �          @�z�@"�\����?У�A\(�C�5�@"�\���H�   ��C��                                    Bxq���  �          @ڏ\@ff�θR>��@|(�C�f@ff��  �ٙ��g�C�O\                                    Bxq��P  �          @�(�@����Q�>�G�@mp�C��@�����ÿ޸R�k�C�b�                                    Bxq���  �          @��H@3�
��?@  @���C�7
@3�
�\�����6�HC�ff                                    Bxq�˜  �          @�
=@[�����@2�\Aď\C�R@[���{?s33A��C��=                                    Bxq��B  �          @׮@U����@%�A���C�8R@U�����?.{@��HC���                                    Bxq���  �          @�@0  ���@��
B&�RC��@0  ��G�@0��A�z�C��R                                    Bxq���  �          @�z�@-p���
=@<(�A�ffC���@-p����?��\AffC�Q�                                    Bxq�4  �          @���@����@��A��
C�B�@��Å>�
=@mp�C�l�                                    Bxq��  �          @���@���I��@$z�A��\C�� @���u�?�{A=��C��=                                    Bxq�#�  �          @�p�@�33�q�?�Q�A��C���@�33���R?��@��C�:�                                    Bxq�2&  �          @�{@r�\��  @,(�A��\C�J=@r�\��(�?k�@��RC��
                                    Bxq�@�  �          @ָR@�G���p�?�p�A�
=C�E@�G���=q>�z�@{C�q                                    Bxq�Or  �          @�z�@�p��Z�H?z�HA\)C��q@�p��c33�u�ffC�aH                                    Bxq�^  �          @�@tz�����@�
A�z�C��@tz���p�>k�?�p�C���                                    Bxq�l�  �          @�p�@Tz���z�@
�HA�G�C��\@Tz����>L��?�  C��                                     Bxq�{d  �          @��H@���+�@=p�A���C���@�����@#�
A�p�C�                                    Bxq��
  �          @Ӆ@��H��Q�@_\)B��C��{@��H�AG�@'�A�
=C���                                    Bxq���  �          @�(�@����=q?�33A�p�C�@����
=>Ǯ@X��C���                                    Bxq��V  �          @�(�@�z��q�?��A{33C��{@�z����>Ǯ@W
=C�z�                                    Bxq���  �          @�33@���i��?��@�=qC��H@���hQ�0�����
C��\                                    Bxq�Ģ  �          @�ff@�(���G�?��Ac�C��f@�(����\���
�.{C��                                    Bxq��H  �          @�33@�=q�Mp�?���Ac\)C�p�@�=q�c33>�p�@S�
C�q                                    Bxq���  �          @�(�@����Z�H?���A�33C���@����x��?#�
@�33C��                                     Bxq��  �          @Ӆ@���hQ�?��RA��HC�N@�����H?(�@��\C��f                                    Bxq��:  �          @��@��R�~�R@
=qA�=qC��f@��R��\)?#�
@��C�7
                                    Bxq��  �          @��H@����Q�@,��A�33C�(�@����{?�ffAz�C��                                    Bxq��  �          @�=q@C33��  ?z�@��HC��R@C33��33���H�N{C�L�                                    Bxq�+,  T          @Ϯ?��H����333�ʏ\C��?��H���
�/\)��
=C���                                    Bxq�9�  �          @��@����=q?�p�AMp�C��@�����þ�z��"�\C�t{                                    Bxq�Hx  �          @�
=@��\���?�\)A`Q�C�"�@��\��33<#�
=uC�1�                                    Bxq�W  �          @ָR@�G��~�R?��HA��C��)@�G����>�G�@o\)C�g�                                    Bxq�e�  �          @�
=@�����=q@(�A�=qC��@������?�@��C��                                    Bxq�tj  �          @�ff@�\)��z�@�\A�Q�C�aH@�\)��?0��@��C��                                     Bxq��  �          @�
=@�\)�|��@+�A�  C�{@�\)��z�?�\)A�C���                                    Bxq���  �          @ҏ\@�ff�fff@8Q�Aϙ�C�G�@�ff��(�?�
=AH  C��                                     Bxq��\  �          @У�@����`  @	��A�=qC�� @�����G�?E�@�=qC��                                    Bxq��  �          @љ�@����H��@\)A��C�J=@����tz�?��HA+
=C��{                                    Bxq���  �          @ҏ\@�����?��A�=qC���@������>��R@-p�C�1�                                    Bxq��N  �          @��H@�ff����?��AB=qC�q�@�ff�����=q�ffC��=                                    Bxq���  �          @��H@�����?��A8z�C�s3@������\)���C��3                                    Bxq��  �          @ҏ\@��
���?˅A`z�C��=@��
���<��
>8Q�C���                                    Bxq��@  �          @У�@�Q��w
=?���A_�C�,�@�Q���p�=�Q�?J=qC��                                    Bxq��  �          @�  @�Q��^�R?\AXQ�C�>�@�Q��s33>.{?��RC�R                                    Bxq��  �          @���@����g
=?��A7
=C�˅@����u��Q�G�C��)                                    Bxq�$2  �          @�33@������B�\��G�C�  @����w
=�33��(�C��R                                    Bxq�2�  �          @ʏ\@�������L�Ϳ�=qC��)@�����G���\���\C��                                    Bxq�A~  �          @ʏ\@�\)���R���R�6{C��R@�\)�n{�0����33C�U�                                    Bxq�P$  �          @���@������\?E�@�ffC���@������\�B�\���C��R                                    Bxq�^�  �          @�@�(��~�R?�p�A��C��@�(���>���@e�C�"�                                    Bxq�mp  �          @�{@����c�
?�z�A�ffC�U�@�����Q�>��H@���C��\                                    Bxq�|  �          @�  @�������?���A�Q�C��=@������>u@�C��)                                    Bxq���  �          @Ϯ@������?�Am�C�
=@�����=�Q�?B�\C���                                    Bxq��b  �          @У�@�  ��p�?���Ac
=C�Y�@�  ��
=����z�C�`                                     Bxq��  �          @У�@��
��{?�{Ag�
C��f@��
�������=qC��=                                    Bxq���  �          @У�@�{�g
=@
=qA�p�C��\@�{��z�?0��@�33C���                                    Bxq��T  �          @�33@���y��@��A��C��H@�����?\)@�(�C��                                    Bxq���  �          @�{@��\(�@333A��
C���@���\)?��A:�HC���                                    Bxq��  �          @�Q�@�p��j�H@A��C�K�@�p�����?Q�@߮C�1�                                    Bxq��F  �          @�\)@�G��w
=@#33A�{C��3@�G�����?s33A
=C�O\                                    Bxq���  T          @�
=@�{��G�@#33A�33C�  @�{��?G�@�
=C�f                                    Bxq��  �          @ָR@���ff@!�A��
C�� @���=q?333@�33C��f                                    Bxq�8  �          @�{@�G��n�R?˅A\z�C�c�@�G�����=�Q�?G�C�B�                                    Bxq�+�  �          @�p�@����u�@A�=qC�Y�@������H?�@��RC��                                     Bxq�:�  �          @�p�@�Q���  @��A��HC�H@�Q���33?=p�@�z�C���                                    Bxq�I*  �          @�@�ff�z�H@)��A�
=C��@�ff���
?�  A	��C��                                     Bxq�W�  �          @�z�@�����  @%�A�ffC�U�@�����p�?fff@��HC�q                                    Bxq�fv  �          @Ӆ@�ff�n{@I��A��C�
=@�ff��(�?��
AVffC��                                    Bxq�u  �          @��H@��H�Z�H@N{A�
=C���@��H��(�?��HAp��C�(�                                    Bxq���  �          @��H@�  �0  @L(�A�p�C���@�  �p  ?�A��HC��
                                    Bxq��h  �          @��
@����c�
@e�B
=C�R@�����z�?�p�A�G�C�N                                    Bxq��  �          @��
@�  �R�\@FffA�p�C���@�  ���R?У�Af�\C�=q                                    Bxq���  �          @��@��R�Z=q@J=qA�
=C�S3@��R��33?��Alz�C��H                                    Bxq��Z  �          @أ�@H����
=@���Bp�C�P�@H������@
=A�\)C���                                    Bxq��   �          @�ff@X���\)@��B  C�1�@X����Q�@ffA�=qC�c�                                    Bxq�ۦ  �          @�p�@Tz����@��B33C���@Tz���=q@��A��RC�H                                    Bxq��L  �          @�(�@a��z=q@xQ�Bz�C�@a����H@ffA�{C�Z�                                    Bxq���  �          @�33@�z��녿
=q��C�k�@�z��33��\)�G�
C�                                    Bxq��  �          @�(�@���Q�z�H�{C���@�녿�=q�������RC�E                                    Bxq�>  T          @�p�@����0  �#�
�\C�G�@����!G�������C�33                                    Bxq�$�  �          @�@��\�{?��A^ffC��
@��\�(��?z�@�p�C��3                                    Bxq�3�  �          @�z�@�{���@�A�=qC�B�@�{�{?��\A8��C�J=                                    Bxq�B0  �          @θR@�
=���
@^�RBz�C�ٚ@�
=�0  @(Q�A�=qC�!H                                    Bxq�P�  �          @�G�@���*=q@\)A��C�{@���Tz�?��A  C�p�                                    Bxq�_|  �          @У�@�p��i��?�G�AV=qC�h�@�p��|�ͼ#�
��G�C�U�                                    Bxq�n"  �          @�Q�@�  �{�?��A>=qC��f@�  ���;�\)��RC�(�                                    Bxq�|�  �          @�  @����{�?���A��\C�U�@������\>#�
?��C���                                    Bxq��n  �          @�Q�@��R����@�A���C�� @��R��?+�@�{C��3                                    Bxq��  �          @�Q�@����A�@#33A��C�� @����q�?�
=A'33C���                                    Bxq���  �          @�G�@����'
=@z�A�Q�C�Y�@����S33?�\)A�\C���                                    Bxq��`  �          @��@�z��*=q@�\A���C�^�@�z��O\)?W
=@�(�C��                                    Bxq��  �          @ҏ\@�G��@  ?��A���C��f@�G��`  ?\)@��C�޸                                    Bxq�Ԭ  �          @��@����J=q?˅Ag�C��R@����a�>k�@C�*=                                    Bxq��R  �          @�z�@��\�e�?�(�AUC�k�@��\�w��#�
��p�C�Y�                                    Bxq���  �          @��
@�=q���R?�=qAAC��@�=q���;�
=�s33C�\                                    Bxq� �  �          @��
@����?�\)A!C���@���  �333���
C�K�                                    Bxq�D  �          @Ǯ@q����?���AffC�R@q����H�h���ffC�                                      Bxq��  �          @ƸR@�G����?��A)p�C��@�G���������HC��\                                    Bxq�,�  �          @ƸR@�p�����?���AD��C�j=@�p����\��
=�xQ�C��=                                    Bxq�;6  �          @�
=@y������?�
=Az�HC�XR@y�����\��  ��\C�u�                                    Bxq�I�  �          @Ǯ@|�����\?��Ad��C�aH@|����녾Ǯ�hQ�C���                                    Bxq�X�  �          @�  @����(�?�(�A�  C��R@�����R�����z�C��3                                    Bxq�g(  �          @���@h�����@Q�A��C�{@h�����>�  @C���                                    Bxq�u�  �          @��@S33���@
=qA���C�c�@S33���>aG�@
�HC��{                                    Bxq��t  T          @���@\����=q?�\)A�z�C���@\����{��G����\C�w
                                    Bxq��  �          @�z�@QG����H?���A�z�C��@QG���p��k��(�C�'�                                    Bxq���  �          @Å@=p����H?�33Ay�C��@=p���=q����p�C���                                    Bxq��f  �          @�(�@J�H����?�(�A^ffC�#�@J�H��{�(����
=C��
                                    Bxq��  �          @��H@,����{?�\)Aw�C�@,�����Ϳ
=���
C�J=                                    Bxq�Ͳ  �          @���@)����{?�p�Ac�
C���@)����33�:�H�޸RC�,�                                    Bxq��X  �          @�\)@W���=q?��AG\)C���@W����O\)��{C���                                    Bxq���  �          @���@HQ�����?��A=C�S3@HQ���33�s33�
�HC�(�                                    Bxq���  �          @�  @���j=q@ ��A�G�C���@�����?Q�@�C�Z�                                    Bxq�J  �          @�p�@�ff�dz�@z�A��HC��f@�ff��ff?.{@ʏ\C�U�                                    Bxq��  �          @\@�=q�H��@(�A�=qC���@�=q�w
=?uA=qC��f                                    Bxq�%�  �          @�=q@�(��>�R@33A�C�N@�(��c33?+�@ə�C��R                                    Bxq�4<  �          @�=q@�p��/\)@��A���C�o\@�p��Y��?fffA��C��                                    Bxq�B�  �          @�p�@����A�@\)A��C�#�@����j�H?O\)@�=qC��\                                    Bxq�Q�  �          @���@�p��I��?�\)A�{C���@�p��h��>�(�@\)C�                                    Bxq�`.  �          @��
@���Y��?У�Aw�C�^�@���qG�=���?s33C��{                                    Bxq�n�  �          @\@����\��?�p�A`��C�3@����p�׽u�
=C���                                    Bxq�}z  �          @�Q�@�Q��Y��?��AO
=C�9�@�Q��i���.{��=qC�>�                                    Bxq��   �          @��R@����Z�H?�z�A[
=C���@����l�ͽ���z�C��                                     Bxq���  �          @�
=@����^�R?
=q@�\)C���@����X�ÿp���G�C�K�                                    Bxq��l  �          @���@�33�i��?���A/
=C�@�33�s33������C�8R                                    Bxq��  �          @�Q�@������u�z�C�Ф@����q녿��
��Q�C�<)                                    Bxq�Ƹ  �          @���@������
�����Mp�C�f@����g
=�   ��p�C��H                                    Bxq��^  �          @���@��
�^�R?�Q�A��C�� @��
�w�=���?h��C��                                    Bxq��  �          @��@���w
=?��RA<��C��=@�����þ�����RC��                                    Bxq��  �          @��H@�  �j=q?��A�
C�0�@�  �qG��\)���C�Ǯ                                    Bxq�P  �          @�=q@����g
=?.{@�  C�|)@����dz�aG��C��                                    Bxq��  �          @�z�@�=q��{?��A+
=C���@�=q���ÿ5��z�C��H                                    Bxq��  �          @��@��H����?8Q�@ָRC��3@��H��{�����)C�H                                    Bxq�-B  �          @�ff@�Q��Mp�?�  A�
=C��@�Q��h��>�  @z�C���                                    Bxq�;�  �          @�@�{�Tz�?��Av=qC��@�{�mp�=���?p��C���                                    Bxq�J�  �          @�@�{�2�\?�=qA�{C��R@�{�S33>��H@��HC��H                                    Bxq�Y4  �          @�@���)��?�(�A�G�C�@���G�>�ff@��RC��                                    Bxq�g�  �          @�
=@��<��?��
A�
=C�%@��[�>\@^{C�>�                                    Bxq�v�  �          @��
@��=q?�AV�HC��@��1�>�=q@%�C�p�                                    Bxq��&  �          @�@�ff�9��?��Aw
=C�j=@�ff�Tz�>�=q@ ��C��3                                    Bxq���  �          @�{@�=q�7�@G�A���C�=q@�=q�\��?�R@�G�C��q                                    Bxq��r  �          @�@�G��;�@�A��C��@�G��aG�?��@�=qC���                                    Bxq��  T          @�ff@�����@�A��C�!H@���5�?�G�A  C�3                                    Bxq���  �          @�ff@�=q��?�Q�A2ffC�~�@�=q�)��=��
?:�HC�H�                                    Bxq��d  �          @Ǯ@��
��?�z�A��
C��\@��
�@��?333@ϮC�XR                                    Bxq��
  �          @�  @����@*�HA�z�C���@���Q�?�z�APz�C���                                    Bxq��  �          @�
=@�=q�=q@��A�G�C���@�=q�E?k�A\)C��                                    Bxq��V  �          @�ff@�(��{@��A��C�  @�(��Q�?�z�A,Q�C��3                                    Bxq��  �          @\@���"�\@ffA�{C�}q@���S33?��AC�>�                                    Bxq��  �          @�ff@�{�z�@Q�A���C�o\@�{�G
=?�z�A4z�C���                                    Bxq�&H  �          @��@����@�A���C�L�@���A�?\(�A=qC�^�                                    Bxq�4�  �          @���@�����R@"�\A��HC���@����Fff?��AR�\C��\                                    Bxq�C�  �          @��@����,��@'
=AѮC���@����c�
?���A;
=C��                                    Bxq�R:  �          @�  @�33�3�
@)��A�G�C�K�@�33�j�H?�
=A5�C���                                    Bxq�`�  �          @��@�=q�Dz�@=p�A��HC�|)@�=q��G�?���AS\)C��
                                    Bxq�o�  T          @�33@�{�AG�@)��A��C�R@�{�w�?��A,��C���                                    Bxq�~,  �          @�p�@�33�6ff@EA�Q�C���@�33�z=q?ǮAv�\C�1�                                    Bxq���  �          @��@x���C�
@E�A��C��@x�����H?��HAg�
C���                                    Bxq��x  �          @�\)@�z��I��@ffA��HC�W
@�z��w
=?B�\@��
C�z�                                    Bxq��  �          @�{@��R�K�@
=A���C�w
@��R�q�?�@��C��                                    Bxq���  �          @��@���>�R@
=A���C�o\@���fff?��@�{C��3                                    Bxq��j  �          @�(�@{��@�A׮C��)@{��G�?�=qAA�C�Ǯ                                    Bxq��  �          @�p�@u��   @�A���C�=q@u��3�
?�p�Ah  C��)                                    Bxq��  �          @�  @��\����@0��B�C�z�@��\�'�?�ffA�
=C��=                                    Bxq��\  �          @��@�녿Tz�@HQ�B  C���@���
=@��AظRC��3                                    Bxq�  �          @�=q@e���\@9��Bp�C�  @e��Fff?��HA��HC��                                    Bxq��  �          @�
=@l���ff@)��A�\)C�Ǯ@l���Q�?���AuG�C�=q                                    Bxq�N  �          @�\)@�Q�޸R?�(�AXz�C��@�Q��ff>���@i��C�
                                    Bxq�-�  �          @��@aG��{@��A�\)C�s3@aG��S33?�\)APQ�C�l�                                    Bxq�<�  �          @��R@��
�*=q?�G�Af{C��3@��
�<(��#�
��C�<)                                    Bxq�K@  �          @�{@���5�?   @�z�C�� @���0�׿L���	��C�1�                                    Bxq�Y�  �          @�p�@n{�tz�?8Q�@�
=C��@n{�p  ����1C�`                                     Bxq�h�  �          @�G�@?\)�vff?У�A�ffC��=@?\)��p��k��%C�˅                                    Bxq�w2  �          @�G�@C33�u?�ffA��HC�<)@C33�����Q�uC��                                     Bxq���  �          @��R@G��Q�@8��B33C��@G���
=?��AC�C�:�                                    Bxq��~  �          @��\?�ff�R�\@4z�B�
C�Q�?�ff��ff?���AR�RC��{                                    Bxq��$  �          @��@
=����?��A\)C���@
=���R�!G���33C�Q�                                    Bxq���  �          @��\>�z�����p���HC�Z�>�zῥ��6ff�C�8R                                    Bxq��p  �          @�(����3�
����K�CixR���
=���#�CC�                                    Bxq��  �          @�(��1G��I����=q�7p�Cd���1G��c�
�����t\)CE�{                                    Bxq�ݼ  �          @�����ÿ޸R�/\)����CJ+����þ�{�N�R���C8z�                                    Bxq��b  �          @�=q�P  �q������RCeY��P  �p��c33�$33CY�                                    Bxq��  �          @�p��1G����\�7���\)Cmk��1G��%��Q��G�RC_
                                    Bxq�	�  �          @�
=�AG������>�R��G�Ci���AG��������FCZ.                                    Bxq�T  �          @�(��c�
�q��-p���=qCb���c�
�
=q����2�\CSE                                    Bxq�&�  �          @�  �I�����ÿ�Q��iG�Ck+��I���Y���Tz��Q�Cc.                                    Bxq�5�  �          @�ff�G����H<�>�{Cv���G���������
=Ct�)                                    Bxq�DF  �          @�{��ff��{�=p���\C{�3��ff�����G
=�=qCxQ�                                    Bxq�R�  �          @�{��(����R=#�
>���Cz�f��(���ff�p��ƸRCx�)                                    Bxq�a�  �          @�=q��\)�y��?B�\A0Q�C��=��\)�tzῊ=q�|Q�C��                                     Bxq�p8  �          @^�R��z��C33?�ffA֏\C�S3��z��[�    <�C��q                                    Bxq�~�  T          @�z�B�\��=q@�A��HC�G��B�\�������@  C���                                    Bxq���  �          @�ff����Q�?�@�ffC�˅����Q������=qC���                                    Bxq��*  T          @XQ�@7������&ff�7�C���@7��}p����R��p�C�u�                                    Bxq���  T          @Vff@G
=�&ff?n{A��RC�\@G
=��G�?�A  C��                                    Bxq��v  T          @�(�@[���z�?��\A��C�t{@[��G�>��@fffC�8R                                    Bxq��  �          @s33@fff�:�H?J=qAB{C�C�@fff��G�>�p�@���C�*=                                    Bxq���  �          @r�\@n{����?
=qA�RC��{@n{��\>�{@�C�R                                    Bxq��h  �          @o\)@`�׼�?�33A���C��q@`�׿��?�G�A|��C���                                    Bxq��  �          @p��@Z=q>L��=#�
?
=@XQ�@Z=q>#�
=�@�@/\)                                    Bxq��  �          @w
=@p  ���
��G���z�C��@p  >B�\������Q�@6ff                                    Bxq�Z  �          @��@��H�
=q�.{�33C�@ @��H���;Ǯ��\)C�0�                                    Bxq�    �          @��@tzῚ�H?^�RAC
=C�8R@tzῼ(�>W
=@?\)C�xR                                    Bxq�.�  �          @���@mp��}p�?�Q�A�C��=@mp���
=?��A	�C�}q                                    Bxq�=L  �          @b�\@\�;�>��@�p�C��@\�Ϳ&ff>B�\@J�HC��{                                    Bxq�K�  �          @`��@]p�=L�;B�\�H��?^�R@]p�>��\)�Q�@{                                    Bxq�Z�  �          @S33@=p��#�
������{C��3@=p�?!G���p����\AA�                                    Bxq�i>  �          @Dz�@Q��\��\�,  C�=q@Q�>��H�33�,�\AO\)                                    Bxq�w�  �          @W
=?����\)�'
=�Wz�C�H?��>aG��5��t��@�{                                    Bxq���  �          @b�\@p��h�����&�C��R@p�>�=q�\)�4�\@��H                                    Bxq��0  �          @
�H?�=q�k�<�?}p�C�l�?�=q�aG����
��C���                                    Bxq���  �          @/\)@p����
�����{C���@p�<���Q��(�?(�                                    Bxq��|  �          @H��@Fff��>B�\@\��C��)@Fff�L��=�G�@   C�0�                                    Bxq��"  �          @N�R@C33��z�?uA��RC�J=@C33�333?5AM�C��f                                    Bxq���  �          @$z�@z�>��?�=qBz�@�
=@z�Ǯ?�ffA�  C���                                    Bxq��n  �          @$z�?��>�ff?���B2ffAc�?����ff?���B2�\C��R                                    Bxq��  �          @^{@  ��33@#33BA�C���@  ��  @B
=C�%                                    Bxq���  �          @���@H�þ�p�@5�B'z�C���@H�ÿ�z�@�B�C�
                                    Bxq�
`  �          @��@(���z�@L(�BQ�RC���@(���  @,(�B'\)C�/\                                    Bxq�  �          @�z�@(Q�=���@Dz�BE�R@��@(Q쿬��@0��B,Q�C�j=                                    Bxq�'�  �          @���?˅@  @n{BU��B[33?˅>u@��HB�k�A	                                    Bxq�6R  �          @�33?�G�?��@��B�33B?�G����\@�p�B��3C��3                                    Bxq�D�  �          @��H?�?��@��Bx��B�?��:�H@��RB�.C�1�                                    Bxq�S�  �          @�ff?��?�@q�Bkp�B\)?����@�  B��C��q                                    Bxq�bD  �          @�33?�{?5@uBzz�A�  ?�{��Q�@n{Bm33C��3                                    Bxq�p�  �          @���@�?+�@fffBk(�A��\@���{@_\)B_�HC���                                    Bxq��  T          @���@���\)@c33B`z�C��@���@@  B2{C�xR                                    Bxq��6  �          @��@0  �\)@Mp�BC�\C�<)@0  �G�@$z�B�C��                                     Bxq���  �          @z=q@
=q�@  @FffBVffC�o\@
=q�Q�@�BQ�C���                                    Bxq���  �          @/\)?��H���R?J=qA�(�C�q�?��H��Q�#�
�aG�C��=                                    Bxq��(  �          @*�H?c�
��>�A7\)C��?c�
�	����R�mp�C�@                                     Bxq���  �          @Dz�?O\)�:�H��(��   C���?O\)����G���RC�j=                                    Bxq��t  �          @%�?L����
�(���o�C��\?L�Ϳ�(���
=�%�RC�z�                                    Bxq��  �          @H��?����R�B�\�eC��{?�녿��ÿ����p�C��q                                    Bxq���  �          @z=q@��Fff��\)���C�.@��%��(���Q�C��\                                    Bxq�f  �          @�  ?�z��C�
��33��ffC��R?�z��33�:=q�<C��=                                    Bxq�  �          @�=q@   �8Q��
=�֣�C�t{@   ��{�Dz��7��C���                                    Bxq� �  T          @Y��@G��z�J=q�\  C�1�@G���z�����C���                                    Bxq�/X  �          @z�H@
=q�H�ÿ
=q� Q�C�AH@
=q�   ��p���
=C�h�                                    Bxq�=�  T          @qG�?�
=�I��>�@���C��{?�
=�?\)��{��
=C�h�                                    Bxq�L�  �          @��@P  �#33>�33@��C���@P  �=q�h���K�C��R                                    Bxq�[J  �          @���@h���(���  �X��C�y�@h�ÿ�ff���\��\)C��
                                    Bxq�i�  �          @��H@\)��Q�k��AG�C�z�@\)���׿��\�X��C�z�                                    Bxq�x�  �          @�
=@�G����R�L�Ϳ0��C��=@�G�����&ff��C��                                    Bxq��<  �          @���@��R�Q녾����vffC���@��R�녿(�����C�`                                     Bxq���  �          @�G�@�  ������UC���@�  ���
�����HC��\                                    Bxq���  �          @��@��;�\)�W
=�*�HC�4{@��;\)���R��=qC�3                                    Bxq��.  �          @��@�{���;�G���p�C�W
@�{>���(���=q?�(�                                    Bxq���  �          @�
=@�
=�\)�W
=�6ffC��@�
=���
��  �Z=qC��)                                    Bxq��z  �          @��@�G��k���\)�y��C�]q@�G���\)��33��  C���                                    Bxq��   �          @|(�@u���(��O\)�>�\C��{@u�<��
�k��W�>���                                    Bxq���  �          @xQ�@l�;�G���ff�{�
C��H@l��>��������
?�p�                                    Bxq��l  �          @l(�@Q녿k������ffC�.@Q녾.{�˅���HC��                                    Bxq�  T          @L(�@��?��Ϳ������RA�  @��?�G��Ǯ��
=B                                    Bxq��  �          @h��@HQ�������
=C���@HQ�>�z��  ��z�@�=q                                    Bxq�(^  �          @w
=@>{�aG��33�G�C��
@>{>�33���@�ff                                    Bxq�7  �          @mp�@-p����R��\��RC���@-p����R� ���*�\C��                                    Bxq�E�  �          @e@=q�z�H��H�,  C���@=q>����%�;G�@��                                    Bxq�TP  �          @��H@:�H�5����{C��=@:�H�{������Q�C�j=                                    Bxq�b�  �          @s�
@��'����
��=qC�l�@���33����  C��q                                    Bxq�q�  �          @o\)@*�H��������G�C�@*�H�������Q�C�N                                    Bxq��B  �          @h��@/\)���
��ff��RC�n@/\)��ff��
�ffC�Y�                                    Bxq���  �          @�  @,�;\)�����C�xR@,��?L�Ϳ��H���
A�p�                                    Bxq���  �          @�=q@c33?��H��Q���ffA���@c33@{�xQ��Q�B 33                                    Bxq��4  �          @���@Q�?�{��z���Q�A�=q@Q�@��}p��dz�B�R                                    Bxq���  �          @��H@c33?�z�����ƸRA���@c33@�B�\�"=qB��                                    Bxq�ɀ  �          @���@c33=u���
���?u@c33?L�Ϳ�ff����AK�
                                    Bxq��&  �          @�{@
�H�˅�L���G�
C��@
�H>L���dz��j\)@�p�                                    Bxq���  �          @�=q?�{�33�e��c��C�9�?�{�#�
��(��HC��                                    Bxq��r  �          @�ff?�녿޸R�o\)�ez�C���?��>��������A7�                                    Bxq�  �          @�?�\)��{�|���~ffC�Z�?�\)?����\).AҸR                                    Bxq��  �          @�33?}p���
=�����v�C��R?}p�>��R��ff�3A��                                    Bxq�!d  �          @�33?�G��XQ쿜(�����C���?�G����.�R�,33C��f                                    Bxq�0
  �          @\)?O\)�U��p���{C��\?O\)��z��XQ��i�\C�s3                                    Bxq�>�  �          @s33?��\�Dz��
=��z�C�>�?��\��=q�>�R�T�HC�aH                                    Bxq�MV  �          @��\@5�*=q���R���C�z�@5��Q�������C���                                    Bxq�[�  �          @��@2�\�'
=������z�C�u�@2�\��{�����RC��                                    Bxq�j�  �          @��H?����R�\�������C���?����Q��8���=��C�5�                                    Bxq�yH  �          @��þ�����G������{C��ᾙ���5�J�H�@{C��3                                    Bxq���  T          @|(�@@�׿z�H����33C��@@��>�\)�����H@��
                                    Bxq���  �          @��@9���ٙ����
�HC��@9�����R�7��2{C��\                                    Bxq��:  �          @�z�@>�R��\)��\��
C��H@>�R����2�\�+�HC��                                     Bxq���  �          @@��@�����
���\��z�C��@�����޸R��C�e                                    Bxq�  �          @]p�@'���{���
���C�S3@'��xQ���\�ffC��R                                    Bxq��,  �          @y��@6ff��R�+�� ��C���@6ff���������C��\                                    Bxq���  �          @���@P���%�z�H�S
=C���@P�׿�  �	�����\C�޸                                    Bxq��x  �          @�@6ff�!G��\)��p�C�>�@6ff����L���9��C��=                                    Bxq��  �          @��@.{�1��   �ٙ�C�*=@.{��33�HQ��6z�C�XR                                    Bxq��  �          @�=q@&ff�)�������
C�<)@&ff��Q��N{�A�C��\                                    Bxq�j  �          @�z�@L(��>{��=q���C���@L(���\�7
=�p�C���                                    Bxq�)  �          @|��@#�
�9�����ffC��3@#�
�{���H���
C��f                                    Bxq�7�  �          @�G�@*=q�1녿�{����C�޸@*=q���
=�\)C��f                                    Bxq�F\  T          @��@N�R����=q����C��)@N�R�����ff��
C�s3                                    Bxq�U  �          @z�H@0���/\)�Ǯ��\)C��)@0���
�H��(�����C���                                    Bxq�c�  �          @|��@p��Dz�=�?��C�Y�@p��-p���Q���(�C��                                    Bxq�rN  T          @~{@\)�E�\)�z�C�p�@\)�%��
=����C���                                    Bxq���  �          @���@H���$z�У����HC�U�@H�ÿ�33�-p���
C��                                    Bxq���  �          @�ff@C�
������p�C�~�@C�
�&ff�HQ��3��C�                                      Bxq��@  �          @�{@%���33�`  �C(�C�� @%�>�33�vff�_�\@���                                    Bxq���  �          @��\@R�\�)���0���\)C���@R�\��Q��
=�ָRC�Ǯ                                    Bxq���  �          @�p�@~{��\?J=qA��C�H@~{������C��                                    Bxq��2  �          @�ff@��H��Q������C��=@��H���׿z�H�M�C��                                    Bxq���  �          @�p�@����=q��p��t  C�f@�����R��{��G�C��
                                    Bxq��~  �          @�(�@����5��  ��Q�C�f@���>����33���@Q�                                    Bxq��$  T          @�p�@���=#�
��
��?
=q@���?��H��(���  A{33                                    Bxq��  �          @��@�ff��  �\)���C�P�@�ff?���G���\)Aa                                    Bxq�p  �          @�Q�@h��    �@���=q<��
@h��?��
�%��HA��R                                    Bxq�"  T          @�ff@dz�>����5���@��H@dz�?�p������\)AΏ\                                    Bxq�0�  �          @��R@_\)�\)�\)�p�C�n@_\)?J=q���z�AJ�H                                    Bxq�?b  �          @j=q@��7
=������{C��f@���
��(�����C���                                    Bxq�N  �          @r�\@�R�%��(���%�C���@�R��׿������C�k�                                    Bxq�\�  �          @��@Q��Z�H?�\)A��C�o\@Q��y���\)��  C���                                    Bxq�kT  �          @�=q@\(���?�ffA�\)C�c�@\(��@��>��R@tz�C�j=                                    Bxq�y�  �          @�z�@�Q��\>�@�z�C�Ff@�Q��\�   ��33C�L�                                    Bxq���  �          @���@�G��>���@�33C��q@�G�� �׿:�H�\)C�xR                                    Bxq��F  �          @��
@�=q��z�0����
C�j=@�=q�\(������}p�C���                                    Bxq���  �          @�33@�Q��ff���H����C��@�Q쿊=q���H�w
=C��                                    Bxq���  �          @���@~�R���;�
=���C���@~�R��\)��ff����C���                                    Bxq��8  �          @�33@l���Q�?�ffAW33C���@l���%����
���C��                                    Bxq���  T          @���@tz��%?�z�Aa�C��@tz��4zᾮ{����C��=                                    Bxq���  �          @�z�@dz��	��?��A��\C�t{@dz��!�=#�
?
=qC�O\                                    Bxq��*  �          @�{@S�
�
�H?�=qA�
=C�aH@S�
�#33<�>�p�C�7
                                    Bxq���  �          @�Q�@h���\)?�  A��HC�+�@h���$zὣ�
���C�b�                                    Bxq�v  �          @���@o\)��H?��A��C���@o\)�/\)�\)��\C���                                    Bxq�  �          @���@w���=q?z�HAN{C�Q�@w�����G���33C���                                    Bxq�)�  �          @���@w
=���?�  AE��C��
@w
=�'
=�������RC���                                    Bxq�8h  T          @�  @XQ��^�R?W
=AG�C��@XQ��Z�H��=q�J�\C�Y�                                    Bxq�G  �          @�  @0������?Q�AG�C�C�@0���w
=��\)����C��=                                    Bxq�U�  �          @�G�@hQ��U�?8Q�Az�C��R@hQ��N{��{�N=qC�1�                                    Bxq�dZ  �          @�(�@`  �Tz�>��R@j=qC�=q@`  �@  ���H��(�C��\                                    Bxq�s   T          @�@A��s�
>�  @:=qC�8R@A��X�ÿ�G����C��H                                    Bxq���  T          @��@Q���ff��{�NffC���@Q��AG��G��Q�C�&f                                    Bxq��L  �          @��R?�33��Q�������RC��?�33�6ff��Q��Jp�C�                                    Bxq���  �          @��R@���ff�W
=�p�C��@��e��J=q�  C��f                                    Bxq���  �          @�p�?�����p��=p���C��?����tz��L(���C��)                                    Bxq��>  �          @��>Ǯ����>W
=@��C�0�>Ǯ���(���z�C�~�                                    Bxq���  �          @�ff�E���ff@
�HA�G�C���E���33�8Q����HC��                                    Bxq�ي  �          @�{>B�\��z�@%A�C�%>B�\��녾L���Q�C�                                      Bxq��0  �          @���?�(����@���Bi�C�Ǯ?�(�����@(Q�A�ffC�8R                                    Bxq���  �          @�G�@9���
=q@QG�B(ffC���@9���c33?�A��C���                                    Bxq�|  �          @�@Q�����>�
=@�
=C���@Q��w���\)���C��                                    Bxq�"  �          @�  @����
>�@��\C�B�@������	����33C�xR                                    Bxq�"�  �          @�\)@>�R���\>�  @'
=C��f@>�R��G�������C�7
                                    Bxq�1n  �          @�@%��Q�>�33@w
=C��\@%�����33��C�]q                                    Bxq�@  
�          @��@>{��{��G��2ffC���@>{�B�\�C33���C�(�                                    Bxq�N�  �          @��@,����  ?333@�=qC�y�@,����ff��G����C�\)                                    Bxq�]`  �          @�=q@����?���A�33C��f@���ff�����=qC���                                    Bxq�l  �          @�{@ff�O\)@g
=B,�C�xR@ff����?���AqC�'�                                    Bxq�z�  �          @��@�\�?\)@y��B8{C��q@�\���H?�G�A��C�C�                                    Bxq��R  �          @�=q@
=����@���Bm  C�+�@
=�u�@O\)B�
C��                                    Bxq���  �          @��H@
�H�u@�G�B��\C�
=@
�H�]p�@r�\B+\)C��                                    Bxq���  �          @��@*�H���@��RBW(�C�g�@*�H�|(�@3�
A�\C�                                    Bxq��D  �          @��@I���%�@n{B)�C�P�@I������?�A�p�C��
                                    Bxq���  �          @���@e��5�@E�B�C��\@e�����?��A>�\C���                                    Bxq�Ґ  �          @�Q�@	���ff@��HBc
=C�Ǯ@	����ff@333A�p�C���                                    Bxq��6  �          @�@<(��g�?ǮA�(�C���@<(��y������{C�~�                                    Bxq���  �          @�
=@\(���z�W
=��HC��)@\(��Dz��8��� ffC�R                                    Bxq���  T          @�\)@[���(�=���?���C�ٚ@[��dz������C��                                    Bxq�(  �          @�33@)�����ÿ}p��*=qC�#�@)���U��N{�z�C�7
                                    Bxq��  �          @���@Dz��w��u�&ffC�7
@Dz��P�����ǮC���                                    Bxq�*t  <          @�33@Z�H�h��?�A���C���@Z�H�����\��(�C��                                    Bxq�9  T          @��
@U�:�H@�A�C�n@U�hQ�>��
@l��C�N                                    Bxq�G�  "          @�@-p����@�G�BD��C�xR@-p���  @�A��
C�\                                    Bxq�Vf  �          @���@{��@��
B\(�C��@{��ff@5�A���C�4{                                    Bxq�e  
�          @��\@*=q���@��
BD{C���@*=q��
=@\)A�33C��                                    Bxq�s�  �          @�=q@8���G�@��BH�HC�|)@8���|��@!�A�z�C��                                    Bxq��X  
�          @��@`���,��@VffB�C�>�@`�����?�Amp�C�n                                    Bxq���  "          @��
@�G��7�@\)A���C�  @�G��g�>Ǯ@{�C��\                                    Bxq���  �          @�p�@���&ff@�A��C�%@���Q�>�33@c33C�!H                                    Bxq��J  T          @���@��
�Q�@p�A�z�C�\)@��
�L(�?�R@���C���                                    Bxq���  
�          @��@����?�Q�A�C�w
@��C�
>\@w�C�o\                                    Bxq�˖  
�          @�=q@w��w
=?}p�A$��C��f@w��s33�����H  C��                                     Bxq��<  "          @�G�@�  �Mp�?���Ar�HC�u�@�  �_\)�����C�L�                                    Bxq���  �          @�G�@�  �J�H?�Q�A�(�C�Ф@�  �mp��#�
�\C��3                                    Bxq���  
Z          @�  @�G��E�?�  A|��C�'�@�G��Z=q��Q��s�
C�Ǯ                                    Bxq�.  "          @�Q�@|(��g
=?=p�@�(�C��H@|(��\�Ϳ���\��C�c�                                    Bxq��  �          @�\)@����g�>�G�@��C��@����S�
��ff��C�c�                                    Bxq�#z  
�          @���@����R�\>��@�z�C��)@����B�\�����b=qC���                                    Bxq�2   "          @���@�\)�Z=q?:�H@�C���@�\)�QG���Q��I�C�%                                    Bxq�@�  T          @���@��R�L(�?
=@��C�7
@��R�AG���Q��G�C���                                    Bxq�Ol  �          @���@���:=q?&ff@���C���@���4z�}p��%C�j=                                    Bxq�^  �          @��\@�(��)��>��@��C��q@�(��p�����/�C���                                    Bxq�l�  
�          @��@�ff�\)>L��@�C���@�ff�{��z��@Q�C��                                    Bxq�{^  
�          @��H@�ff�?\)>�=q@4z�C��@�ff�+���{�ap�C�*=                                    Bxq��  �          @�(�@���:=q����\)C�c�@�������z����C��H                                    Bxq���  �          @�(�@����1G�<�>�33C�B�@����
=�����m�C�                                      Bxq��P  �          @��@�(��)���L����RC�@�(���������z�C��=                                    Bxq���  T          @���@n�R�c�
@G�A��HC�/\@n�R���\�B�\��Q�C�5�                                    Bxq�Ĝ  
�          @���@g��n{@{A�
=C�)@g����\��G�����C��3                                    Bxq��B  T          @��@o\)�g�@�A���C���@o\)��p��.{��Q�C���                                    Bxq���  "          @��H@u�`  @z�A��
C�ٚ@u��녽�G���33C��{                                    Bxq���  "          @�
=@n�R�l��@G�A���C���@n�R���H�L�Ϳ�C�Z�                                    Bxq��4  �          @���@X���|(�@#�
A�p�C�\)@X����ff=u?
=C��                                    Bxq��  
�          @��@o\)�z=q?�=qA��C��)@o\)���ÿ
=q���RC���                                    Bxq��  �          @�ff@N�R����@3�
A�(�C�E@N�R��>aG�@
=C��)                                    Bxq�+&  T          @�{@]p�����@*=qA���C�Y�@]p���=q=���?s33C���                                    Bxq�9�  
�          @��@��\�l(�?޸RA�{C��@��\�����   ����C��q                                    Bxq�Hr  �          @�p�@�\)�(��?�@��HC�f@�\)� �׿xQ��G�C���                                    Bxq�W  �          @�  @�33��\>�33@aG�C���@�33��z�J=q��{C�+�                                    Bxq�e�  �          @�
=@����
�H>�=q@+�C��3@��ÿ�p��n{��RC��)                                    Bxq�td  �          @�p�@��\� ��>��@'�C��
@��\��׿�\)�6�HC���                                    Bxq��
  �          @�p�@���>���@W
=C��@��
=�p�����C��                                    Bxq���  "          @���@�{�
�H>k�@ffC��=@�{���H�xQ���C��\                                    Bxq��V  T          @��@�(��>���@��HC���@�(���Ϳh�����C�k�                                    Bxq���  �          @�@�{�\)>Ǯ@~�RC�W
@�{�ff�^�R���C��)                                    Bxq���  
�          @��@���(�?fffAffC�g�@������
=���C��H                                    Bxq��H  
�          @���@�33��33?���A^=qC��f@�33��=�?��HC��)                                    Bxq���  �          @��@����{?L��AC��@���� �׿&ff���HC��R                                    Bxq��  T          @�z�@��
�"�\?�  AM��C�<)@��
�3�
�����E�C��q                                    Bxq��:  T          @�z�@����H��?�\A��C��@����fff�B�\��Q�C�                                    Bxq��  
�          @��@���R�\@��A��HC���@���z�H=#�
>�
=C�(�                                    Bxq��  
�          @�\)@���5@S33B�C��=@�����?�G�AD(�C�&f                                    Bxq�$,  �          @�Q�@�G��
�H@dz�B33C���@�G��p  ?���A���C�s3                                    Bxq�2�  S          @���@tz��@�B1Q�C���@tz��w
=@\)A�33C�T{                                    Bxq�Ax  
�          @���@n�R�+�@��B%�HC�*=@n�R��{?��RA�\)C�f                                    Bxq�P  
Z          @��H@|���>{@vffBz�C��=@|����=q?�33AtQ�C�n                                    Bxq�^�  �          @θR@����n{@��A�
=C�q@������þB�\��p�C�                                      Bxq�mj  �          @θR@�����@z�A�z�C���@�����R��  �p�C���                                    Bxq�|  T          @θR@�����  @&ffA�
=C���@�������=#�
>���C��                                    Bxq���  �          @�
=@�Q���33@
�HA��HC�޸@�Q����
��p��Q�C�%                                    Bxq��\  "          @Ϯ@���dz�@��A�G�C��
@������=���?c�
C�>�                                    Bxq��  �          @�  @����k�@
=qA�\)C���@������׾#�
���C�w
                                    Bxq���  
�          @�ff@���`��@�\A�  C�8R@����{=#�
>�p�C���                                    Bxq��N  "          @�Q�@���U@�A�
=C�N@�����H>W
=?���C���                                    Bxq���  �          @�@�\)�L(�@ ��A�  C���@�\)����>���@eC�|)                                    Bxq��  �          @�  @����P��@!G�A��HC�}q@������>�p�@S33C�Z�                                    Bxq��@  �          @�G�@�p��`  @!G�A�
=C�B�@�p����>u@z�C�b�                                    Bxq���  "          @�=q@�\)�c�
@��A�G�C�:�@�\)��G�=�G�?s33C��H                                    Bxq��  T          @��@�Q��k�@'
=A�G�C�%@�Q���Q�>W
=?��C�J=                                    Bxq�2  �          @�G�@�G��k�@ ��A��HC�8R@�G����R>\)?��HC��f                                    Bxq�+�  T          @љ�@�  �w
=@0��A�C��@�  ���>�  @
�HC��H                                    Bxq�:~  "          @�p�@�=q��G�@8Q�AծC���@�=q���R>��@33C���                                    Bxq�I$  �          @�p�@~{���@{A���C�%@~{��  ��=q��HC�4{                                    Bxq�W�  �          @��@�
=���\@�A��C���@�
=���׿z����HC���                                    Bxq�fp  T          @��H@�����@33A��C���@���(��aG�����C��                                    Bxq�u  T          @ҏ\@�
=�~{@�A�  C��
@�
=��=q�aG���C���                                    Bxq���  
�          @ҏ\@����~�R@=qA�{C���@��������G��xQ�C�|)                                    Bxq��b  
�          @�p�@z=q�`  @j�HB�
C��@z=q��?��HA-��C�8R                                    Bxq��  �          @�p�@x���}p�@K�A�=qC�9�@x������?
=q@�33C�˅                                    Bxq���  
Z          @�{@w
=���@EA�  C��R@w
=��(�>Ǯ@_\)C�y�                                    Bxq��T  
�          @�z�@dz����@S�
A��\C���@dz���  ?z�@�{C�"�                                    Bxq���  
(          @�@z=q��z�@%�A���C��R@z=q���\�B�\���HC���                                    Bxq�۠  �          @�\)@u���G�@7
=AӅC��@u�����=���?^�RC�O\                                    Bxq��F  T          @�p�@�(���Q�@5�Aљ�C��=@�(����>aG�@   C��                                    Bxq���  T          @��@W
=����@hQ�B
\)C���@W
=���?^�R@��C��                                    Bxq��  �          @�(�@S�
�w�@tz�B��C�H�@S�
��=q?��A$z�C��{                                    Bxq�8  �          @��@333�k�@{�B!p�C��f@333��
=?�=qAJ{C�                                      Bxq�$�  �          @�Q�@C33��(�@]p�B�C�9�@C33��33?0��@�(�C���                                    Bxq�3�  �          @���@}p�����@�RA�\)C�N@}p���=q��G�����C���                                    Bxq�B*  �          @��H@�����?�\A�C��@����(��\(����C�AH                                    Bxq�P�  �          @�  @�����=q@  A�  C�@�����zᾙ���*=qC�.                                    Bxq�_v  �          @���@�  ���
?�
=A�z�C�@�  ��Q�����33C�w
                                    Bxq�n  �          @ʏ\@�z����?��AC�
C�n@�z���zῇ���HC�&f                                    Bxq�|�  �          @��
@��\��=q?�(�Az�HC�.@��\����@  ��  C�7
                                    Bxq��h  �          @�z�@�=q��33?��
A��C�3@�=q����5���C�f                                    Bxq��  �          @���@������?�=qA`z�C�w
@�������h��� Q�C��\                                    Bxq���  �          @Ϯ@����ff?�\)AC\)C�>�@�����׿�{�C��q                                    Bxq��Z  �          @��H@��|��?˅AiG�C���@���{�J=q��p�C�
                                    Bxq��   T          @��@�(�����?�33AM��C�o\@�(���p���  �{C�f                                    Bxq�Ԧ  �          @��@����ff?��AL  C�s3@����G������   C�'�                                    Bxq��L  �          @��@�����z�?�
=AQ��C�˅@�����Q쿃�
��\C�e                                    Bxq���  �          @�Q�@�\)���
?��HAW�
C��\@�\)��Q�}p��=qC�8R                                    Bxq� �  �          @ȣ�@������?��RA\��C�P�@����{�h���C��R                                    Bxq�>  �          @Ǯ@�Q����H?���AL��C��@�Q���{����C��\                                    Bxq��  �          @Ǯ@�(����?\AaC�8R@�(���=q�xQ���HC���                                    Bxq�,�  �          @�Q�@�\)��
=?�{A$(�C���@�\)���
��(��Z{C��                                    Bxq�;0  �          @ȣ�@����z�H?�ffAd��C��{@�����(��O\)��z�C�0�                                    Bxq�I�  �          @�
=@�����G�?��AG\)C�
@�����(���ff�(�C���                                    Bxq�X|  �          @�Q�@����s�
?��AM�C��@����}p��c�
�=qC�%                                    Bxq�g"  �          @��@�{���\�����z�C��{@�{�W
=����p�C�t{                                    Bxq�u�  �          @�z�@�G���z�>��?�\)C�/\@�G��qG��\)��\)C�U�                                    Bxq��n  T          @\@�{��33?��@�(�C���@�{�~�R����p�C�<)                                    Bxq��  �          @\@��R����?J=q@�Q�C���@��R�u��G��f�HC��f                                    Bxq���  �          @��@�  ����?�\@�=qC�)@�  �j=q�޸R���C�p�                                    Bxq��`  �          @��@��H��\)�W
=�   C��
@��H�\���p����C��q                                    Bxq��  �          @�\)@�33���þaG���
=C��3@�33�l���(Q���  C���                                    Bxq�ͬ  �          @ƸR@�(��~�R?=p�@�p�C���@�(��p  �\�e�C�}q                                    Bxq��R  �          @�G�@����u?��
AG�C��f@����r�\�����7�C��                                    Bxq���  �          @���@�p��mp�?aG�A
=C�@�p��fff��  �@(�C�+�                                    Bxq���  �          @��H@�G��dz�?�z�A/
=C���@�G��hQ�s33�(�C�h�                                    Bxq�D  �          @Å@�\)�c�
?�p�AaG�C���@�\)�r�\�+��ȣ�C��                                     Bxq��  �          @���@��H�W
=?�(�A��
C���@��H�p�׾�p��[�C�
                                    Bxq�%�  �          @�p�@�z��J=q@G�A��HC���@�z��p      �#�
C�@                                     Bxq�46  �          @ə�@���<��@
�HA��C��q@���j=q>u@�C�33                                    Bxq�B�  T          @�
=@���3�
@-p�A�(�C���@���u�?E�@��C���                                    Bxq�Q�  �          @���@����*=q@:�HA�
=C��@����tz�?��\A\)C�k�                                    Bxq�`(  �          @�Q�@����+�@#33A���C���@����hQ�?333@�\)C�)                                    Bxq�n�  �          @�G�@�
=�0  @,(�A���C�� @�
=�qG�?J=q@�{C�ff                                    Bxq�}t  �          @�  @��Ϳ�ff@mp�B�RC��)@����8��@#�
AȸRC��                                    Bxq��  �          @θR@��z�@{A��
C�o\@��R�\?W
=@�Q�C�h�                                    Bxq���  �          @�Q�@z=q��  @�G�B6C�y�@z=q�dz�@333A�33C��                                    Bxq��f  �          @�z�@l�Ϳ�G�@��RBI{C���@l���fff@R�\B  C���                                    Bxq��  �          @Ǯ@l�Ϳ�ff@�  BG{C��)@l���w
=@J=qA��RC��                                    Bxq�Ʋ  �          @��
@w
=���@�G�B@ffC��)@w
=�a�@HQ�A�
=C��                                    Bxq��X  �          @�\)@�녿�Q�@�{B7��C�9�@���g
=@>{A噚C�0�                                    Bxq���  T          @���@z=q���@�\)BCC�� @z=q�hQ�@R�\A�\)C���                                    Bxq��  �          @�=q@~{��(�@�p�B?ffC��q@~{�p  @I��A�  C�P�                                    Bxq�J  �          @���@�Q쿰��@�(�B1�C�f@�Q��aG�@<��A��C�4{                                    Bxq��  �          @ʏ\@��H�xQ�@��B5{C��@��H�N�R@P��A���C��f                                    Bxq��  �          @ə�@�����R@��RB(��C���@���aG�@0  A�C���                                    Bxq�-<  
�          @ə�@�����
@{�BG�C���@�������@z�A�z�C�U�                                    Bxq�;�  �          @��
@�\)�333@p��B=qC�@ @�\)��z�?�Au�C��q                                    Bxq�J�  �          @��
@�=q�E�@^�RB(�C�>�@�=q��
=?��
A:�\C�H                                    Bxq�Y.  �          @���@����G
=@b�\B�HC�\@�������?���A>�\C��                                    Bxq�g�  
�          @θR@�Q��Vff@K�A�{C���@�Q�����?aG�@��C�l�                                    Bxq�vz  
�          @�@��R�\)@0  A�
=C�B�@��R���H>8Q�?˅C�}q                                    Bxq��   �          @�Q�@��R��
=@P  A�\)C�Z�@��R�XQ�?��HA~{C��H                                    Bxq���  "          @��H@�
=��Q�@�33B{C�Ff@�
=�L��@4z�A�=qC��
                                    Bxq��l  "          @Ӆ@�p��ٙ�@j�HB�C��@�p��Y��@(�A�p�C�T{                                    Bxq��  �          @��H@��R�E�@[�A���C�e@��R��{?��RA-�C�U�                                    Bxq���  �          @ҏ\@�
=�J�H@O\)A�G�C��@�
=���?��
Ap�C�y�                                    Bxq��^  �          @ҏ\@�z��
=@��A�G�C��f@�z��QG�?B�\@�z�C���                                    Bxq��  
�          @��@�{��@&ffA�G�C���@�{�333?�ffA8��C��                                    Bxq��  
�          @�{@�����p�@;�A�z�C��R@����4z�?�
=Aip�C��                                    Bxq��P  �          @�ff@����
@@��A֣�C��{@���dz�?�ffA4Q�C�~�                                    Bxq��  �          @�z�@����@$z�A�33C���@��K�?��\A�\C�c�                                    Bxq��  �          @��@��
�(��@7�A��HC��@��
�p��?�G�A  C�}q                                    Bxq�&B  T          @�(�@�����>�(�@l(�C�B�@�����
������C�N                                    Bxq�4�  �          @�33@Å�(�<#�
=L��C�)@Å��
��ff�6�\C���                                    Bxq�C�  �          @�33@�ff�H��>�Q�@L��C��3@�ff�6ff��\)�BffC���                                    Bxq�R4  T          @�ff@�33�z�H@P  A�  C��)@�33����?&ff@�G�C�`                                     Bxq�`�  �          @�{@�����  @�A���C��@��������
�0��C���                                    