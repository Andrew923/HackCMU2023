CDF  �   
      time          *   Conventions       ACDD-1.3, Spase v2.2.3     title         /DSCOVR Magnetometer Level 2 One Minute Averages    id        Doe_m1m_dscovr_s20230221000000_e20230221235959_p20230222021649_pub.nc   naming_authority      gov.noaa.swpc      program       DSCOVR     summary       }Interplanetary magnetic field observations collected from magnetometer on DSCOVR satellite - 1-minute average of Level 1 data      keywords      _NumericalData.ObservedRegion.Heliosphere.NearEarth, NumericalData.MeasurementType.MagneticField    keywords_vocabulary       Spase v2.2.2   
references        �; DSCOVR TIME SERIES DATA AVERAGES ALGORITHM THEORETICAL BASIS DOCUMENT, v2.4; GSE TO GSM COORDINATE TRANSFORMATION ALGORITHM THEORETICAL BASIS DOCUMENT v2.1      metadata_link         �http://www.ngdc.noaa.gov/docucomp/page?xml=NOAA/NESDIS/NGDC/STP/Space_Weather/iso/xml/satellite-systems_dscovr.xml&view=getDataView&header=none    license       Spase.Access Rights.Open   institution       NOAA   source        DSCOVR Magnetometer Level 1    platform      'Deep Space Climate Observatory (DSCOVR)    
instrument        +boom-mounted triaxial fluxgate magnetometer    history       ,DSCOVR real-time telemetry processing system   	algorithm         FDSCOVR MAGNETOMETER LEVEL 1B DATA ALGORITHM THEORETICAL BASIS DOCUMENT     algorithmVersion      B      algorithmDate         
2015-10-15     processing_level      Level 2    processing_level_description      11-minute average using Hodges-Lehmann M-estimator      date_created      2023-02-22T02:16:49.561Z   date_calibration_data_updated         2023-01-28T00:00:00.000Z   time_coverage_duration        P01D   time_coverage_start       2023-02-21T00:00:00.000Z   time_coverage_end         2023-02-21T23:59:59.000Z   time_coverage_resolution      PT1M   creator_name      Doug Biesecker     creator_type      person     creator_institution       DOC/NOAA/NWS/NCEP/SWPC     creator_email         doug.biesecker@noaa.gov    creator_url       http://www.swpc.noaa.gov/      publisher_name         National Geophysical Data Center   publisher_type        institution    publisher_institution         DOC/NOAA/NESDIS/NGDC   publisher_email       william.rowland@noaa.gov   publisher_url          http://www.ngdc.noaa.gov/dscovr/   records_maximum         �   records_present         �   records_data        �   records_fill             records_missing                    time                description       "date and time for each observation     
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
short_name        overall_quality    C_format      %d     units         n/a    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale        7lBxgQ�  �          @�(���ff�r�\���H�^{C{
=��ff��{��ff�CB��                                    Bxg`f  �          @��Ϳ��H�u���33�[�HCy.���H��p���\)�RCB(�                                    Bxgo  �          @�  ���R�e�����iQ�Cz�῞�R�#�
���
 L�C4@                                     Bxg}�  �          @�=q��ff�n{�Å�b�Cw^���ff�\)����u�C9
=                                    Bxg�X  T          @���=q�l(���\)�`�Cv�{��=q�8Q������ C:�{                                    Bxg��  �          @��
��G��aG���Q��f{Cv�Ϳ�G����
�޸R�{C4��                                    Bxg��  �          @����p��b�\�����f33Cw=q��p����߮
=C4�R                                    Bxg�J  T          @��H��=q�`  ��33�Yp�Cu���=q��{��z���C@                                      Bxg��  �          @ٙ���
=�n�R���H�OQ�Cr����
=�
=q�Ϯ�RCC��                                    BxgՖ  �          @Ϯ�B�\�L�����
�9Q�Cbp��B�\��G���33�u
=C<J=                                    Bxg�<  �          @�G���z�xQ��l����C>�q��z�?�G��l(����C(�{                                    Bxg��  �          @�
=��=q��{�p  �*{C8�=��=q?˅�Z=q��C��                                    Bxg�  �          @�Q��l(��z���ff�N33C<��l(�?�{����:(�CE                                    Bxg.  �          @�  ���ÿW
=�����>33C?!H����?�z������2�HC�                                     Bxg�  T          @�(���{���\��33�;
=CA  ��{?�����R�4(�C �{                                    Bxg-z  �          @�
=���R��z�����:��CA+����R?��H��Q��4Q�C!                                    Bxg<   �          @������H������33�9Q�C7.���H@#33��  � �RCu�                                    BxgJ�  �          A{��(�>�33�\�AC0T{��(�@U���H��C!H                                    BxgYl  �          A�����\?(�����Fz�C-����\@k������
=C\)                                    Bxgh  �          A33��
=?Q����R�2�C+����
=@dz��������C                                      Bxgv�  "          AQ���{�L����{�/  C4���{@333��ff�
=C                                    Bxg�^  �          A�\��녿���p��D�C:!H���@%���
�.ffC��                                    Bxg�  �          A{���Ϳ!G��˅�E�C:������@(����=q�0ffC��                                    Bxg��  �          A�
�������R�߮�_p�CL������?�\�ᙚ�bG�C�)                                    Bxg�P  �          A(����H����=q�K\)CF.���H?������Ip�C !H                                    Bxg��  �          A\)������G����H�X(�CB�3����@�����Kp�C#�                                    BxgΜ  �          A�
�g��8Q���Q��bp�CZ� �g�?�G���Q��}C$h�                                    Bxg�B  �          A	p��p  �.{��ff�d�CW��p  ?��R���H�z
=C!�q                                    Bxg��  �          A
{��Q��5���V�CU�R��Q�?�  ���m��C&�)                                    Bxg��  �          A(���녿#�
��\)�SffC;+����@3�
�����;  C�q                                    Bxg	4  
�          A���Ӆ@��\)�ffC"z��Ӆ@�G��5����C�\                                    Bxg�  T          A=q���
@W��8Q����HC�����
@���Q����
C�=                                    Bxg&�  T          A��@Mp��>{����C�R��@�Q�z�H�ָRC{                                    Bxg5&  T          A����G�@Dz��hQ��ϙ�C�3��G�@�
=��\)�4(�C��                                    BxgC�  "          A�R�ڏ\@E�tz���{C��ڏ\@��\���
�C
=C(�                                    BxgRr  T          A\)��33@G��u����C�=��33@����\�B{C\                                    Bxga  "          A�H��
=@QG��\(���ffC�{��
=@���������C�)                                    Bxgo�  �          A33��z�@*�H�h����Q�Cz���z�@�(���=q�HQ�C�3                                    Bxg~d  �          A�\���@*=q�I�����\C ����@w�����C
                                    Bxg�
  
�          A{��(�?�ff�\����33C%����(�@Q��ff�o�
CJ=                                    Bxg��  �          A
=����?�Q��X���ģ�C$�\����@W���p��`��CǮ                                    Bxg�V  T          A=q��G�@(��P  ����C ޸��G�@n�R��{�6�HC\                                    Bxg��  �          A�޸R@(���N{����C=q�޸R@x�ÿ�p��(��C��                                    BxgǢ  T          A ����(�@2�\�<(�����C���(�@xQ쿕��\C�
                                    Bxg�H  �          @��
��\)@���*�H���
C h���\)@Y�������Q�C.                                    Bxg��  �          @����ָR@ ���*�H��\)Cz��ָR@`  ������Cn                                    Bxg�  �          @�
=��33@&ff�:=q���C���33@l(����R�p�C�\                                    Bxg:  T          @�33��ff@AG���Q��lz�C��ff@e��B�\��Q�C�H                                    Bxg�  T          @�=q��33@.{���H�8z�C����33@E=��
?#�
C�                                    Bxg�  �          @����
=@_\)��33�"�HCh���
=@n{?�\@k�C�H                                    Bxg.,  �          @�=q��{@�.{���C!^���{@XQ쿚�H��\C
=                                    Bxg<�  �          @�R��33@-p��aG��ٙ�Cc���33@333?z�@�ffC��                                    BxgKx  �          @�Q����H@@�׿aG�����C�����H@C33?8Q�@��\C��                                    BxgZ  �          @�Q�����@5����H�4(�CT{����@L(�>�?uC�
                                    Bxgh�  �          @�ff���
@!G�����%��C�H���
@6ff=��
?��Cz�                                    Bxgwj  �          @�ff��\)@0�׿�\�N�RC
��\)@Q녾8Q쿨��C�)                                    Bxg�  �          @�z����@!녿�\)�D(�C +����@@  �#�
��p�Cٚ                                    Bxg��  �          @�\)���
@�
���R�g33C"�{���
@?\)�
=����C�f                                    Bxg�\  �          @���˅@Q녿��R�;�
C�˅@e�>��
@�RC�)                                    Bxg�  �          @�G�����@�p��aG���ffC	)����@�
=?�33AQC
O\                                    Bxg��  �          @����R@��\��G����C  ���R@��?�\)A/\)C��                                    Bxg�N  �          @�\)���@�\)��(���C�����@���?�G�@�=qCs3                                    Bxg��  �          @��
����@�G��z�H��C�����@�{?���A'33C��                                    Bxg�  
Z          @����@��
���\��HC����@��R?h��@��HC��                                    Bxg�@  "          @�(���  @G��\)��=qCL���  @E��W
=��(�C�)                                    Bxg	�  "          @�
=���@~{�xQ���HC.���@{�?��A
=Cff                                    Bxg�  �          @���Q�@#33�   ��C�f��Q�@L�Ϳ   �{�C�{                                    Bxg'2  T          @�ff�ҏ\@%����H�w\)C���ҏ\@Mp���G��\��C�                                    Bxg5�  �          @�����R@Vff�   ��ffC�����R@�z�
=q��\)C33                                    BxgD~  �          @�R��p�@X���=q���HC=q��p�@�(���G��Z=qC5�                                    BxgS$  �          @���Q�@�  ��ff�>=qC� ��Q�@��R?8Q�@�Q�Cff                                    Bxga�  T          @�����@b�\�z�H���C������@�  ��Q��UCE                                    Bxgpp  "          @�R��G�@qG����q��C\��G�@�\)>#�
?�  C
=                                    Bxg  �          @�p���  @W
=>��?���C����  @7
=?�\A`(�CE                                    Bxg��  �          @�{��@n�R>u?��C�
��@H��@�A}�C�3                                    Bxg�b  �          @�
=��\)@c�
?&ff@���C5���\)@2�\@33A��\C�                                    Bxg�  �          @���{@E?�G�A8��C:���{?�(�@4z�A�{C#��                                    Bxg��  �          @����H@
�H?B�\@��\C"�����H?��
?�(�AU�C'��                                    Bxg�T  "          @��
����@��*=q����CO\����@W
=���G�C��                                    Bxg��  �          @�\���@P  �!���33C  ���@�=q�!G�����C@                                     Bxg�  �          @��
��Q�>�����R�P  C1�R��Q�@1���ff�*�
C޸                                    Bxg�F  T          @��
�|��@G���
=�P�HC�)�|��@��������  C�R                                    Bxg�  �          @�\����@H�����H��C�����@�
=�   ���
C
=                                    Bxg�  �          @��H��p�@hQ���ff��C�{��p�@��\�Q���
=C��                                    Bxg 8  �          @�\)��G�@Q���G��33C���G�@�{�'
=��ffC(�                                    Bxg.�  �          @�R�c33@�G��l(�����B�B��c33@�ff�L�����
B�=                                    Bxg=�  �          @��
�h��@��
�s33���B����h��@�=q�\(���{B��H                                    BxgL*  T          @��H�qG�@�Q��L(���B�8R�qG�@��;�  ��p�B�                                    BxgZ�  �          @���(�@\)�U��G�C�q��(�@�33��  � ��C��                                    Bxgiv  "          @�����=q@w
=����\)Cs3��=q@�����=q��
C�R                                    Bxgx  
�          @�\�ƸR@Fff�33���Cz��ƸR@l�;������C0�                                    Bxg��  �          @�\���R@Q��%���
=C)���R@��
�.{��Q�C33                                    Bxg�h  �          @ᙚ��\)@\���{���HC�3��\)@��H��z��z�CxR                                    Bxg�  
�          @�Q�����@|(���G��YG�Cff����@���>�33@(��C\                                    Bxg��  �          @����
@�{�Q���=qC�����
@�p�>�=q@�C
��                                    Bxg�Z  �          @�����R@~{�:=q���C(����R@�(��&ff���Cs3                                    Bxg�   
�          @���G�@g���Q��3\)C���G�@w�>�@k�Ch�                                    Bxgަ  
Z          @�33���
@`  ��33�(�C!H���
@g�?+�@�G�CJ=                                    Bxg�L  �          @�p���=q@Dz���I�C���=q@_\)<#�
=#�
C�H                                    Bxg��  
�          @�(��ٙ�@
=q�Q�����C"h��ٙ�@A녿���p�C�R                                    Bxg
�  �          @����H@(Q�?�
=A��\C�����H?��@<��A�C$��                                    Bxg>  �          @�\)�qG�@N{@\BKG�Cz��qG����
@�(�Bu(�C5.                                    Bxg'�  "          A�R���@[�@�33BB�HC8R���=���@߮Bn{C2��                                    Bxg6�  "          A�Å@y��@eA���Cs3�Å?��H@���B\)C$T{                                    BxgE0  	�          A ����p�@���@��BQ�C޸��p�?�G�@�Q�BM�\CO\                                    BxgS�  �          AQ���ff@�z�@�=qA�p�C��ff@
=@��B7�
C��                                    Bxgb|  
(          A  ��(�@�  @��B�C
����(�?�\@�ffBC(�C!                                      Bxgq"  T          A{��  @�  @�Q�Bp�C

=��  ?�@ÅBC��C )                                    Bxg�  �          A z���@�  @��
B33Cٚ��?��
@�z�B`C޸                                    Bxg�n  �          A   ��Q�@�  @���B=qCh���Q�?���@��B]Q�C�R                                    Bxg�  T          @����33@���@\(�A��C�f��33@G�@�  B�RC �3                                    Bxg��  "          A(�����@�G�@�\)B&�C�����?s33@׮B[G�C(h�                                    Bxg�`  "          A����
@y��@�=qB*�C�{���
?L��@�Q�B\��C*:�                                    Bxg�  
�          @�\)���@�  @�p�B33C�q���?�@�33BT=qC"�R                                    Bxg׬  �          A (���  @�{@l(�A�z�C���  @
�H@�33B(�C��                                    Bxg�R  �          AG���p�@�p�@<��A��
C���p�@<(�@�B��C�\                                    Bxg��  �          AG���(�@��@L��A�\)C�q��(�@<��@��RB!Q�C:�                                    Bxg�  �          Ap���Q�@��@z�A�z�C���Q�@hQ�@��\B
��C�
                                    BxgD  �          A(�����@���?E�@��HC������@��R@aG�A�Q�C
5�                                    Bxg �  �          A��=q@��R��Q���
C
\��=q@��?�ffAQ�C	�H                                    Bxg/�  �          A���\)@�G��Mp���Q�C	aH��\)@��R���O\)C�)                                    Bxg>6  �          A���ff@|(��z�H��Q�Cp���ff@|(�?z�H@��Cp�                                    BxgL�  �          A Q�����@9����{���CB�����@L��>.{?�Q�CE                                    Bxg[�  
�          A   ��  @�R�n{��
=C!����  @(Q�>�{@��C �R                                    Bxgj(  �          A ����R@b�\�O\)��  C�{��R@`��?p��@�
=C
=                                    Bxgx�  �          AG����H@|�Ϳ�  �C33C�����H@��>���@�\C�R                                    Bxg�t  �          A�H���H@�Q��
=q�m��C�����H@����#�
��  CL�                                    Bxg�  �          Az����@��J�H���\C  ���@��Ϳ:�H��{C�{                                    Bxg��  �          A�H��G�@p  �.{����C�R��G�@��H�(����\)C�                                    Bxg�f  
Z          Aff��\@&ff�J�H��G�C z���\@q녿˅�.�RC�q                                    Bxg�  
�          A����=q@=q�G���=qC!���=q@e����5��C�f                                    Bxgв  
�          A��陚@B�\�;�����CaH�陚@�����
=� z�C�                                    Bxg�X  �          A	��=q@���?(��@�{C����=q@z�H@3�
A�
=C�                                    Bxg��  �          A���љ�@�
=?�z�A Q�C���љ�@w�@UA�p�Cu�                                    Bxg��  �          A�R��  @�(�@G�A_
=C{��  @X��@��A�Cu�                                    BxgJ  �          A���  @���@  A{�
C
����  @hQ�@�{BffC�\                                    Bxg�  �          A\)����@�  @�RA�
=C	Ǯ����@`  @�(�B	��C�\                                    Bxg(�  T          Ap����@�{@p�A�G�Ck����@]p�@��HB=qCB�                                    Bxg7<  
�          A�����@�p�@=p�A���C
  ���@N{@�Q�B(�C��                                    BxgE�  T          @�\)��{@��
@��RA�p�C���{@�@�\)BB��C#�                                    BxgT�  "          A���@�\)@5A�G�C
:����@U�@�{B�CaH                                    Bxgc.  �          A���  @��\�p����RC޸��  @�zὣ�
�(�Cu�                                    Bxgq�  
(          AQ�����@��H�\)�z�HC=q����@x��?���A{Cu�                                    Bxg�z  �          A����\@~{<�>L��C����\@`��?���AMG�Ch�                                    Bxg�   T          Az���@��H��\)���C0���@p��?У�A1�C+�                                    Bxg��  T          Ap���@�Q쿗
=�=qC����@��?J=q@��
CT{                                    Bxg�l  T          A���{@�z�
=q�o\)C\��{@{�?�A��CW
                                    Bxg�  
�          A �����@Vff�+�����CB����@R�\?p��@�{C��                                    Bxgɸ  �          Az��θR@��?˅A/�
CQ��θR@mp�@l(�A��HC�                                    Bxg�^  �          A�����@�(�?��\@�\)C=q����@xQ�@H��A�33C=q                                    Bxg�  �          A�R��
=@���?   @Y��C.��
=@�  @/\)A�{C�3                                    Bxg��  �          A{���H@���@�\Ab{C�{���H@c33@��A�C�                                    BxgP  �          Ap�����@�\)@9��A���Cp�����@G�@�33B33C��                                    Bxg�  �          A������@�Q�?���A33C�f����@w�@\��A�ffC�
                                    Bxg!�  T          Ap��أ�@��H=��
?��Cn�أ�@���@G�A}G�C�R                                    Bxg0B  T          A����@��H?#�
@�p�C���@���@0��A�{C��                                    Bxg>�  �          A  ���@��?���A
=CǮ���@l(�@VffA�  C��                                    BxgM�  
�          A=q����@��H@�A�33CaH����@2�\@�=qA�\C�
                                    Bxg\4  T          A=q�ڏ\@�G�@p�At��C���ڏ\@5�@z=qA��HC��                                    Bxgj�  �          A�ָR@���?�33AS
=C�
�ָR@L��@p  A�=qCz�                                    Bxgy�  �          A�����H@�?��
@�(�C.���H@|(�@H��A�=qC#�                                    Bxg�&  T          A�H��  @�>L��?���C����  @��@��A��RCn                                    Bxg��  
�          @�����@��ÿG���z�CB�����@��H?���A7\)CW
                                    Bxg�r  I          @�
=��@�
=��p��I��C��@�
=?0��@�  C
�f                                    Bxg�  �          A Q���z�@��R������
CQ���z�@�녽�\)�   C	�3                                    Bxg¾  
�          @������@��\�Dz���33C�����@�\)�0�����RCz�                                    Bxg�d  �          A   ���H@�G��^{��C�����H@��������p�C�H                                    Bxg�
  
�          @����{@�(��n�R��Q�CaH��{@�녿��\�{C ��                                    Bxg�  
(          @��R�xQ�@�\)��  �&{C��xQ�@�33�,(����B��f                                    Bxg�V  
�          A ���Vff@����  �933B�  �Vff@�
=�L�����B�3                                    Bxg�  
�          Aff�tz�@��H����(��CǮ�tz�@У��333���\B�R                                    Bxg�  T          A\)��Q�@�p���Q��C
�{��Q�@\��\����C��                                    Bxg)H  
�          A���n�R@}p��ȣ��DG�CO\�n�R@ʏ\�vff��p�B���                                    Bxg7�  �          A���
@�����  �d�B�G���
@�
=��\)�(�B�(�                                    BxgF�  �          A��\@n�R��p��f�HB�{��\@�������
=B�B�                                    BxgU:  �          A�H��\@����\)�S��B�\��\@�\)�i�����B��                                    Bxgc�  
�          A��?�@������h��B��?�@�z���=q�   B�B�                                    Bxgr�  T          A�>��H@���
=�B�Ǯ>��H@�\)���A\)B���                                    Bxg�,  
�          A�
?��H?�G���\)�qBL?��H@��
�Ϯ�Q{B�{                                    Bxg��  �          A(�?�=q?�p���z�L�B`\)?�=q@�Q���ff�C\)B��\                                    Bxg�x  T          A  @G�?��\��33#�B ��@G�@�33��\)�P�B��\                                    Bxg�  �          AQ�?�\)?Y�����.A�(�?�\)@����ٙ��^{B��
                                    Bxg��  "          A�?���#�
�   ��C��?��@XQ���  �x�HBx{                                    Bxg�j  T          AG�?Y����Q��\)¨\C��?Y��@a���p��B�Ǯ                                    Bxg�  
�          A�>��>�(��
=«��B7ff>��@��\�����s��B�p�                                    Bxg�  �          A  ?�?L���G� ��B	?�@��
��z��f�B���                                    Bxg�\  "          A
�R?�p�>L���{=q@�33?�p�@vff��ff�o\)Bz�H                                    Bxg  
Z          A	G�?�{�aG������C���?�{@[�����z�BvQ�                                    Bxg�  �          A�
?���?�z���R�B��?���@��\�׮�R=qB���                                    Bxg"N  	�          A��@
=q?�{���B�@
=q@�
=��=q�I��B�\                                    Bxg0�  T          A
=q@1G�?�Q�� Q��=A��@1G�@�����33�H=qBo�                                    Bxg?�  
�          A33@E�@33����{B��@E�@�����{�5�Bn�R                                    BxgN@  "          A
=q@:�H@=q��{�{B��@:�H@�  ��(��-p�Bx
=                                    Bxg\�  �          A33@W�@;����l{B$
=@W�@�����(�Bpp�                                    Bxgk�  "          A�@333@
=q���z�B\)@333@�Q���
=�4\)Bw�
                                    Bxgz2  
�          A33?�(�?�Q���G�B�?�(�@��������Pp�B�#�                                    Bxg��  �          A
=?�?���=q�\B Q�?�@�G���33�Z{B�u�                                    Bxg�~  T          @�(�?�{���
���HffC��f?�{@P  �ۅ�up�Bp��                                    Bxg�$  T          A{@��?��R���u�A�{@��@�  ����P{Bu�\                                    Bxg��  "          A��@.{?�
=��ff�)A��H@.{@�=q��\)�Az�Br�                                    Bxg�p  
�          @�z�@}p��xQ���(��i�\C��@}p�?�(�����]�A�
=                                    Bxg�  T          A ��@^�R����G��qffC�,�@^�R?�33��33�u�A���                                    Bxg�  "          AG�@ff����p�C�5�@ff?������B��                                    Bxg�b  T          @�\@
=�!G���G�aHC���@
=?!G���R\A�(�                                    Bxg�  "          @��?У��5���
�~�
C��f?У�>����ff=qA(�                                    Bxg �  T          @��
?�ff�4z���33=qC�XR?�ff>�p�����\)A|z�                                    Bxg T  �          @�\)?��Ϳ�33��
=�=C�y�?���?�z���3333B8                                      Bxg )�  
�          A   ?\��G���p�C��?\?����\z�BL��                                    Bxg 8�  �          @�
=@@�׾W
=��
=�3C��q@@��@7���(��g��B.                                    Bxg GF  �          @�G�@�\)�.{��\)�2�HC�
@�\)?��R�����*�
A�33                                    Bxg U�  T          @�  @��?�(��w��Q�A�G�@��@L(��1���G�A��                                    Bxg d�  "          @�
=?�33?������
8RBF�R?�33@�p�����>
=B�                                      Bxg s8  �          A�H�#�
?
=�=q«��B�\)�#�
@�=q���t�B�\                                    Bxg ��  T          A���
>.{��
±��B�Lͼ��
@g�������RB��3                                    Bxg ��  "          AQ�����H��
­.C�33��@AG���{\B�(�                                    Bxg �*  
�          A����=q�!G�� z�ª8RCw  ��=q@333���(�B��                                    Bxg ��  �          A�?
=q�E��¦�qC���?
=q@,����{�\B�B�                                    Bxg �v  "          A�H@
=?�G����R�3A��@
=@��
��
=�Q��Bv�                                    Bxg �  �          A�?�  ?
=q���=qA���?�  @u���G��lp�B���                                    Bxg ��  �          A33?�33?G�� �� u�B	{?�33@�=q��\)�kQ�B��\                                    Bxg �h  �          A�?Y����
�ffaHC�,�?Y��?\�Q�33Bsff                                    Bxg �  "          A��?�33�7���p��C��?�33?z���\¢�A�p�                                    Bxg!�  
�          A z�?�33�!��ۅ�HC�*=?�33?����G� B�A�(�                                    Bxg!Z  T          A��=��
��=q�����=
=C�g�=��
�#�
��G��\C��                                     Bxg!#   �          Aff>�G�@[����H�y(�B�� >�G�@�33��=q��B�u�                                    Bxg!1�  T          A�R?�\)@Y����  �|��B�  ?�\)@�33��{��B���                                    Bxg!@L  �          A(�?��@�
=����LG�B�\?��@�(��l������B�Ǯ                                    Bxg!N�  �          A�H�O\)@�(��[���z�B�#׿O\)A�<�>8Q�B�{                                    Bxg!]�  T          A�
��  @�\)�_\)��33B�(���  A z�L�Ϳ�B̏\                                    Bxg!l>  �          A��z�@ᙚ�[���Q�B�aH�z�@��H�k���=qB��                                    Bxg!z�  
�          Aff��
@�=q�����BӀ ��
@�p�?J=q@��B��                                    Bxg!��  "          A33���
@�\)�p��|(�B��쿣�
@��R?�33A=qB�L�                                    Bxg!�0  
Z          @�(����@��H���V=qB�
=���@�p�?���A+33B��                                    Bxg!��  "          @��R�h��@�z���J�\B�
=�h��@�?��RA4��B���                                    Bxg!�|  
�          @�\)>��H@ҏ\�z�H���
B�z�>��H@��fff�ָRB���                                    Bxg!�"  
Z          @�녿u@��Ϳ&ff���HB�aH�u@�  @!G�A�{B�(�                                    Bxg!��  T          @�z����@���\�l��B�aH����@��@(Q�A���B�
=                                    Bxg!�n  �          @�(��8Q�@�z�?���A*�RB����8Q�@�{@�\)B�B���                                    Bxg!�  
�          @�
=� ��@�
=?�z�AG�B�\� ��@�@�ffB�B�ff                                    Bxg!��  
�          @�G�����@�G�@:�HA�(�B�\)����@���@�p�B9��Ḅ�                                    Bxg"`  z          @�=q��ff@��R@o\)B�RB�{��ff@P  @���B`\)B��
                                    Bxg"  
�          @��
��Q�@C�
@�(�BQ\)B�B���Q�?��
@��B��RCB�                                    Bxg"*�  
�          @�p���p�@���@O\)B=qB�B���p�@Q�@��HBjQ�B�                                    Bxg"9R  
�          @љ����\@��H�-p���C8R���\@�녿=p��θRC ٚ                                    Bxg"G�  �          @�(���{@�(��HQ���(�C���{@�  ��G���C �f                                    Bxg"V�  "          @�  �p��@��H�0�����
B��R�p��@�G��
=����B�\)                                    Bxg"eD  �          @���^{@�(��@����G�B�.�^{@���333���
B��f                                    Bxg"s�  
Z          @����@^�R�˅�az�B�=q���@�\)��{��B�                                     Bxg"��  �          @���   @N{�˅�effB����   @�������{B���                                    Bxg"�6  "          @�=q�C33@����(��*33B��H�C33@�{�/\)���\B�G�                                    Bxg"��  
�          @�=q�1G�@������3�HB����1G�@����@  ��=qB�G�                                    Bxg"��  �          @��
�3�
@�Q���(��%��B�\�3�
@ə��%���\B�                                      Bxg"�(  �          @�G��{@�
=�����5B�(��{@�Q��G
=����B�ff                                    Bxg"��  "          @��
�@��@�=q�Z=q��p�B��H�@��@Ǯ���
�33B�k�                                    Bxg"�t  .          @޸R�l(�@�G��{��\)B��l(�@��\�8Q쿹��B���                                    Bxg"�  
�          @�\)�h��@�p�����z�B����h��@�=q=��
?333B�                                    Bxg"��  T          @�p��r�\@�
=��p��U�B����r�\@���?
=@���B���                                    Bxg#f  "          @ə���(�@�����R�9�C����(�@��>��H@�=qCxR                                    Bxg#  �          @�{�z�H@�33�k��
=C �3�z�H@�=q?�ffA��C�                                    Bxg##�  
4          @��
�Vff@��׿fff��ffB� �Vff@�?�=qA<��B�Q�                                    Bxg#2X  
N          @��R��(�@��������B��)��(�@���@
=A}C �)                                    Bxg#@�  �          A{��ff@��ÿ\)�|��B�����ff@θR@Q�As
=B�(�                                    Bxg#O�  T          A\)��p�@�33��\)��\)B�p���p�@��@=qA���B�                                    Bxg#^J  
�          A�����@޸R�h����
=B��=����@أ�?�\)AL��B�
=                                    Bxg#l�  
Z          Az���=q@�(��k���ffB�W
��=q@�?�APQ�B��
                                    Bxg#{�  
�          A
{��ff@�\)�����33B�ff��ff@�\)?��A�HB�W
                                    Bxg#�<  
�          A���@����(Q����B�Q���@���>��?�G�B�\                                    Bxg#��  "          Ap�����@����\)�
\)B�������@�ff���p��B�Ǯ                                    Bxg#��  T          A
�\�l��@�����
�!z�B����l��@�\)�Dz����B�.                                    Bxg#�.  
(          A	��o\)@�Q���33�4��C 8R�o\)@׮�s33��B�#�                                    Bxg#��  �          A ���P  @���������B���P  @�z����~�HB�                                     Bxg#�z  �          A Q��Z=q@����{��=qB����Z=q@�녿����8Q�B�\                                    Bxg#�   
�          @�{�S�
@��H����33B�  �S�
@�z��.{��G�B��                                    Bxg#��  T          @�z��Z�H@�z�����3�
B����Z�H@��]p��У�B��f                                    Bxg#�l  
�          @�(��J�H@�����
=�(�B��)�J�H@�p��?\)��p�B�z�                                    Bxg$  �          @�\)�U�@�G���=q�p�B�\)�U�@�\)�(���  B�                                    Bxg$�  "          A�R�S�
@����\�33B�\)�S�
@��H��
��\)B���                                    Bxg$+^  "          @���?\)@\�q���B�W
�?\)@�녿�Q����B��)                                    Bxg$:  �          A�\�~{@��R��{��B�
=�~{@�p�� ������B��                                    Bxg$H�  �          A����  @�
=��Q���B�\)��  @ۅ����B�u�                                    Bxg$WP  "          A ����ff@���vff��
=B�.��ff@�ff��
=�#�
B�=q                                    Bxg$e�  �          @�33�p��@�=q��G��W
=B�u��p��@У�?@  @�
=B��                                    Bxg$t�  "          A���=q@�(���(��^{C�
��=q@�p�>�@XQ�C L�                                    Bxg$�B  
�          A�
���H@ָR=L��>�Q�B������H@�ff@$z�A��B�{                                    Bxg$��  �          A�H�w�@�p��������B����w�@��H�!G����
B���                                    Bxg$��  
Z          @�p���33@��\�=q��  B�\��33@�33�u���HB�                                    Bxg$�4  
�          @�����@�  �{��=qB�������@��þu��\)B���                                    Bxg$��  
�          @�p��dz�@.�R��33�B�C���dz�@��n{���C �
                                    Bxg$̀  �          @�R�g
=�����{�rffC:n�g
=@G���(��`
=CǮ                                    Bxg$�&  	.          A
=�p��L����z��uQ�Chff�p��k�� ����C9Q�                                    Bxg$��  �          A���=q@(Q���ff�K�C����=q@�\)������Cff                                    Bxg$�r  T          A�����@fff��=q�7{C8R���@��������CO\                                    Bxg%  �          A�����@�z�����#  C������@�Q��z�H���C�                                    Bxg%�  �          Az����H@fff��{�*C�q���H@�G���{���HC�3                                    Bxg%$d  �          A	����G�@8Q���p��-�HC����G�@�33��{���HC
�                                    Bxg%3
  �          A������@^�R����  C�����@����|(��ՅC	z�                                    Bxg%A�  
�          A\)��{@5��p��"�
CxR��{@�
=��\)��33C�\                                    Bxg%PV  "          A{��\)@|�����R�  C����\)@��\�Y�����RC�R                                    Bxg%^�  �          A33��z�@�����{���C�H��z�@�z��P�����C�R                                    Bxg%m�  �          A33��=q@Q�����%
=C!^���=q@��
����Q�C�                                    Bxg%|H  �          A�����?\��
=�&�
C&�f����@e��Q��C5�                                    Bxg%��  
Z          Az��ҏ\?�����ff�%G�C&J=�ҏ\@j=q��
=���C�                                    Bxg%��  
Z          A���Q�?�
=��\)�-{C)����Q�@W
=��(��(�C��                                    Bxg%�:  �          A����Q�?������!  C/��Q�@!���ff�\)CǮ                                    Bxg%��  T          A�H��Q�?8Q���G���Q�C.�q��Q�@�����R����C"�)                                    Bxg%ņ  "          A���=q=u���H���C3�=��=q?����
=����C%��                                    Bxg%�,  �          A\)��=q����=q���HC7޸��=q?��R��{��33C*�q                                    Bxg%��  "          A���ff�n{��
=��RC:���ff?h����\)��
C-@                                     Bxg%�x  "          A�R�����H��33��C?����?#�
��Q���\C.�)                                    Bxg&   
�          A�H����33�����HCCxR�������z���C5
=                                    Bxg&�  
�          A
=��R�33��ff�	�
CE�R��R��������RC6aH                                    Bxg&j  T          A=q����"�\���\�   CG.��녿&ff��p���HC9�                                    Bxg&,  �          A���p��-p���p��	G�CJ#���p��J=q����z�C:                                    Bxg&:�  
Z          AQ��ҏ\�'���Q��ffCI���ҏ\�   �\�*�\C8Y�                                    Bxg&I\  �          Az���  ��\)����G(�CC����  ?}p����K��C*B�                                    Bxg&X  T          A���׿�(������JQ�CEG�����?aG����P�C*�R                                    Bxg&f�  
�          A������B�\���Z�RC<h����?�������R  C��                                    Bxg&uN  �          Aff�����O\)��Q��6��CU� ���Ϳk���Q��Vz�C>�f                                    Bxg&��  �          A�����  �����Rp�CNB����>k���Q��c�
C1#�                                    Bxg&��  �          A
=��{�(�����P��CL�q��{>�z���  �`�\C0�                                    Bxg&�@  T          A(���33�����Ӆ�L�CEn��33?O\)��Q��R\)C*�R                                    Bxg&��  �          A=q��Q쿐���޸R�a�CB!H��Q�?�\)��p��_{C#!H                                    Bxg&��  
�          Aff��{�xQ���=q�e�C@W
��{?�ff�޸R�_��C �R                                    Bxg&�2  
Z          AQ���Q쿇��陚�l
=CB  ��Q�?����
=�g��C {                                    Bxg&��  
(          A�H��(��E��߮�l  C>�H��(�?ٙ���=q�c=qC��                                    Bxg&�~  
�          A (���Q�<��
��p��o�C3�=��Q�@=q�Ϯ�X�
C�q                                    Bxg&�$  
Z          A   �R�\�=p���  �C@���R�\?�����w��C.                                    Bxg'�  
M          A��)����z����H
=CW��)��?=p�����C$k�                                    Bxg'p  "          A
�H�z�H�b�\���HC~���z�H�!G��	G�¤�{CT��                                    Bxg'%  
�          A
�R�h���hQ������C녿h�ÿ:�H�	G�¤�CZ��                                    Bxg'3�  
�          A
=q�@  �s�
��\)�}(�C�h��@  �p���	�¤Cg#�                                    Bxg'Bb  
Z          A
=�(�ÿ�Q���H�HC{@ �(��?xQ����£��B��                                    Bxg'Q  "          A  ���,(�� ��W
C�ff��>aG���¬=qC�3                                    Bxg'_�  
�          A�H���{� z�\C����>���ffª��C�f                                    Bxg'nT  �          Azᾔz��<(����\)C�.��z�L���(�°C>n                                    Bxg'|�  T          A
�R��\)�I����.C�xR��\)�k��
�\¯=qC[.                                    Bxg'��  
�          A	���  �Y�����
#�C����  �
=q���¬{Cu                                      Bxg'�F  
�          A�
��Q��`  ��
=�C�þ�Q�0���33©p�Cr��                                    Bxg'��  T          A�>#�
�������op�C�\>#�
���
���
=C��                                    Bxg'��  �          A��?&ff�����p��c��C��?&ff��
=���ffC�W
                                    Bxg'�8  �          A��5���\��=q�c
=C��3�5��Q��33�3Cy��                                    Bxg'��  
�          A	>�Q����
�ָR�L��C��>�Q��4z���z�C���                                    Bxg'�  
�          Aff�u��ff��33�O(�C����u�,(����R��C�\)                                    Bxg'�*  T          A�H>��R������33�CffC���>��R�G
=���H�\C���                                    Bxg( �  
�          @�\)?���ᙚ=u?�\C�.?����p���\��C��f                                    Bxg(v  �          @�\)?޸R���>��
@Q�C��q?޸R���H�
=q��  C��f                                    Bxg(  
(          @�Q�?���陚=�Q�?5C�{?����p�������C�Z�                                    Bxg(,�  �          @�Q�?����Å��\)��C��?��������Ǯ�YffC���                                    Bxg(;h  "          A�Ϳ�{�e�����{z�C|�\��{�u�G�ffC]{                                    Bxg(J  
Z          @��H����?Y����(�33C)����@I����ff�xffB�W
                                    Bxg(X�  "          A"�\�^�R?�(��   ¢��B��3�^�R@�{����}�B�{                                    Bxg(gZ  
�          A�׿���?�\�33¥Q�C=q����@e��(��B�G�                                    Bxg(v   T          A���^�R��p����¨��CK(��^�R@1G��=q�RB���                                    Bxg(��  T          A\)�   ��R�{ªW
Cf�R�   @�R����Bʽq                                    Bxg(�L  �          Ap��c�
?�  ��£\C�H�c�
@w���\�B��
                                    Bxg(��  
�          A33�B�\?L���{¦B�C\)�B�\@n{��(�B��                                    Bxg(��  �          A�׿�R������
«{CO�3��R@'
=�	k�Bγ3                                    Bxg(�>  �          A��G��p����¤aHCfLͿG�?��R�33k�B޽q                                    Bxg(��  T          A\)���ÿ�  �Q��{C_ff����?�33�33�C �H                                    Bxg(܊  �          A�R��녿��H��HL�CXs3���?�p��p���CaH                                    Bxg(�0  �          @�ff��  �fff��R�\CS
��  ?����33\)C+�                                    Bxg(��  �          @ۅ��{��z���Q�B�C\J=��{?}p�����C��                                    Bxg)|  T          @��H�h���ff��=q\)Cv���h�þ�z���
= ��CExR                                    Bxg)"  
�          @����{<#�
��z�¢L�C3�f��{@�\�أ��qB�Ǯ                                    Bxg)%�  
�          @�zΎ��#�
��{ 33C4}q���@(��陚(�B�=                                    Bxg)4n  "          @���G�?��R����p�C:ῡG�@l(��ə��i  Bٽq                                    Bxg)C  �          @���=q?����ۅ��C
��=q@l����  �]�
B虚                                    Bxg)Q�  �          @�G����>�����G��C)�����@(Q�����B���                                    Bxg)``  T          @���R    ��\)ª(�C3���R@�\�ۅ�)B�k�                                    Bxg)o  �          @�(�?333�n{���H�p\)C�Z�?333������p�u�C���                                    Bxg)}�  �          A�?G���{��33�b�RC�H?G��33�����C�q�                                    Bxg)�R  �          A ��?�  ���������O��C�S3?�  �#33��RaHC�                                    Bxg)��  �          @�  ?�Q���z��:=q��{C���?�Q���p���{�)�
C�(�                                    Bxg)��  "          @��?�=q�Ǯ�5���
C���?�=q��G������&��C�^�                                    Bxg)�D  
Z          @��?(�����@  ��z�C�Ф?(���p������/��C��                                    Bxg)��  
g          @�  >�G����
�hQ���33C�*=>�G���ff�����G  C���                                    Bxg)Ր  �          A��?�����H����{{C��?�녿�(��ffz�C���                                    Bxg)�6  	�          A{?�����ff���v=qC��\?��Ϳ�{��H�3C�4{                                    Bxg)��  
�          A
�\?�\)��\)����m��C���?�\)��Q��\)�C�}q                                    Bxg*�  T          A
{?��������R�offC�l�?����ff����C�xR                                    Bxg*(  "          A��?xQ���ff��
=�g
=C�+�?xQ��   ��
�RC��                                    Bxg*�  T          A�R?G�������p��\G�C���?G��=q� ��{C�                                      Bxg*-t  �          @��<������9
=C�(�<��[���=qG�C�AH                                    Bxg*<  �          Aff�������׮�U(�C�J=���.{��ff8RC���                                    Bxg*J�  �          @�{�n{������
�?ffC��3�n{�L(���{ffC}�3                                    Bxg*Yf  �          @��R����{�+�����C������Z=q�z=q�B33C���                                    Bxg*h  �          @���?�\)���H��{�k�C��=?�\)��ff�1G�����C��)                                    Bxg*v�  �          @���?c�
����P���
=C�XR?c�
�a���G��M�C�)                                    Bxg*�X  T          @���>�������&�
C���>��K������l�HC�@                                     Bxg*��  
�          @�33?��\��z��Q��ҸRC��\?��\�l���j�H�,�C��H                                    Bxg*��  
�          @��?����
=�fff��RC�xR?���<(���ff�^C��\                                    Bxg*�J  �          @�Q�>aG���G��:�H�Q�C��
>aG��>{��  �U=qC�(�                                    Bxg*��  �          @���u����S�
�Pz�Cx5ÿu����xQ�\Cic�                                    Bxg*Ζ  
�          @k����8���ff�G�C��f���Q��7��R��C�!H                                    Bxg*�<  �          @�z᾽p��U��33�

=C��᾽p��\)�L���O�C��                                    Bxg*��  �          @�  ����Q��N�R�2(�C��{����
�H����w�C�
=                                    Bxg*��  T          @}p�=��ͿL���s33
=C���=���>�p��w�¨�qB�z�                                    Bxg+	.  �          @Q녾8Q��{�333�o�\C�޸�8Q�5�I���HC�q                                    Bxg+�  "          @�녿:�H�c33�[��-�RC�,Ϳ:�H����=q�p��C|�)                                    Bxg+&z  �          @��ÿ
=q��p��3�
��ffC�"��
=q���H��G��1��C�o\                                    Bxg+5   "          @�(���\)���
��z����C��
��\)��\)�2�\�z�C���                                    Bxg+C�  T          @��H�B�\��z`\)�Y��C�\�B�\����>{��C��\                                    Bxg+Rl  /          A �׿B�\�\)��z�¨C>\)�B�\@���\\Bڅ                                    Bxg+a  G          A(��z�?��(��HB��Ϳz�@�\)� ���r�
B��)                                    Bxg+o�  	.          A{��@*�H�
�RB���@��H��Q��^��B��3                                    Bxg+~^  �          A�
��  @!G�����
B�W
��  @�\)��{�bG�Bʳ3                                    Bxg+�  
�          A(��E�?�����¢�B��f�E�@o\)�ff�B�k�                                    Bxg+��  
�          A33�xQ�@
=�p�.B���xQ�@�����
�k=qB�                                    Bxg+�P  
�          A'���33@?\)��H8RB���33@���
=q�^p�B�Q�                                    Bxg+��  T          A(�ÿ�
=@1G�� ��33B�=��
=@�����bffBճ3                                    Bxg+ǜ  "          A(z���
@!G�� ���B�z���
@���ff�g=qB�                                    Bxg+�B  T          A'33��Q�?�  �#\)�qC^���Q�@�z���R33B�L�                                    Bxg+��  
�          A+33�\@Q��%���B�
=�\@������rp�BոR                                    Bxg+�  T          A,zῪ=q?��R�)� ��C	  ��=q@�\)��ffB�Ǯ                                    Bxg,4  �          A*�H�L��?
=q�*{©�{C{�L��@c�
� z�  B�u�                                    Bxg,�  a          A+��\)>����+33­C�H�\)@W��"�\��B�                                    Bxg,�  �          A,�ÿ\)�L���,Q�­��C9B��\)@AG��%p��fB��                                    Bxg,.&  "          A,zᾳ33���,Q�°\CF����33@<���%  B��{                                    Bxg,<�  �          A,z�Ǯ��Q��,Q�¯�RCA�Ǯ@>{�%���qB�\                                    Bxg,Kr  "          A,�Ϳ#�
��Q��,z�­.C<T{�#�
@=p��%�k�B�=q                                    Bxg,Z  "          A,  �.{�#�
�+33ªC_�H�.{@���'33�B�ff                                    Bxg,h�  �          A+\)�aG��W
=�)�¦��C_ٚ�aG�@(��&�H�=Bߞ�                                    Bxg,wd  �          A+
=�fff�c�
�)¦p�C`���fff@Q��&�H  B��                                    Bxg,�
  �          A*�\�xQ�5�)��§
=CX&f�xQ�@�\�%�  B���                                    Bxg,��  T          A(�ÿ8Q쿐���'\)¥aHCm^��8Q�?��%\)B��H                                    Bxg,�V  
�          A%���333�:�H�$��¨ǮCbxR�333@
�H�!p��Bי�                                    Bxg,��  
�          A(�ͿY���&ff�'�¨\)CY#׿Y��@33�$  \Bܙ�                                    Bxg,��  "          A)p��u��=q�%k�Cruÿu?�\)�'�£��Cz�                                    Bxg,�H  
�          A(Q�h������#33�Cy33�h��?
=q�'\)¨z�C.                                    Bxg,��  
�          A'\)���H�(Q�� ���Cu\)���H>���&=q¦u�C'�
                                    Bxg,�  
�          A)������&ff�"�H�Cu.����>��R�(  ¦� C%��                                    Bxg,�:  
Z          A*�\��
=�   �$���Ct��
=>�G��)G�¦u�C�R                                    Bxg-	�  �          A)녿���.{�#33Cw:ῑ�>B�\�(��§z�C*k�                                    Bxg-�  
�          A)���{�C33� ��8RCy�R��{�.{�((�§ǮC<�                                    Bxg-',  T          A&�R�����Mp������C{{���;Ǯ�%��§#�CGp�                                    Bxg-5�  T          A&{�s33�J=q�G��HC}:�s33��33�%�¨CHL�                                    Bxg-Dx  �          A$�׿�ff�b�\����C}����ff�J=q�#33¥\)CX��                                    Bxg-S  "          A$(����
�i������3C~@ ���
�h���"�R¤��C]��                                    Bxg-a�  �          A$Q�=p��`�����C���=p��B�\�#\)¨{Ca�R                                    Bxg-pj  
Z          A$�þ��
�Mp��Q�z�C�'����
��G��$z�­�Cj�                                    Bxg-  "          A&=q�����l���33�C�o\���ÿp���%p�©  Cz��                                    Bxg-��  
�          A(�ÿ\)��  �  C�  �\)��(��'�¥aHCu=q                                    Bxg-�\  "          A)G������
��ǮC��;������'�¤ffCzk�                                    Bxg-�  �          A(��=�Q���33��=qC��{=�Q����%�aHC�k�                                    Bxg-��  �          A)����
���H�33�t�C�1쾣�
���#�C�>�                                    Bxg-�N  �          A'��W
=��{�G��p�C�Ф�W
=�$z��"=qp�C��)                                    Bxg-��  
�          A$��=���  ��
Q�C�ٚ=������#�¥=qC��3                                    Bxg-�            A&�\��\)��33����=C��H��\)��z��$��¤k�C��                                    Bxg-�@  �          A)>W
=��z��ff�y��C�@ >W
=�{�%��qC��                                     Bxg.�  "          A(�ͼ��
��ff�
=�~�C��ἣ�
���%p���C���                                    Bxg.�  T          A'���\)�����\��C��{��\)�����$���C�H                                    Bxg. 2  �          A&�H��R�����  �y�C�T{��R�
�H�"�H\C~#�                                    Bxg..�  "          A(�����������H�pG�C}s3�����G��W
Cn��                                    Bxg.=~  T          A�����������Q��jz�C|�׿���z��
�HQ�Co�                                    Bxg.L$  
�          Aff���\��=q��G��U��C��R���\�N{��H\)C|}q                                    Bxg.Z�  
5          A�׾B�\�U��H�C�Z�B�\�c�
��
©{C���                                    Bxg.ip            A�\�\)�z=q���fC����\)��z��Q�¢aHCx8R                                    Bxg.x  
Z          A�׿+���p��	G��33C�h��+��������CyW
                                    Bxg.��  
�          Aff�G���Q���p��p  C�
�G���\��Q�C{0�                                    Bxg.�b  "          A  ������  ��{�U  C��쿐���L������Cz��                                    Bxg.�  
�          A�ÿ��\���R��
=�@\)C��{���\�u��\)�{{C.                                    Bxg.��  
g          A=q��G����H��=q�U��C�����G��P���33C|�
                                    Bxg.�T  T          Aff��Q�������z��D=qC���Q��u���\�}z�CyY�                                    Bxg.��            A�
��=q�Q��ff��C{���=q�@  ��R¤�)CV�f                                    Bxg.ޠ  T          A%G���p����� Ce�)��p�>�\)�"{��C+��                                    Bxg.�F  
�          A&=q�=q�xQ���
�)CI޸�=q?У���\G�C                                      Bxg.��  	�          A%��:�H�h���\)��CET{�:�H?�����ǮCG�                                    Bxg/
�  
�          A$Q��@  ����RCS���@  ?0����=qC'�                                    Bxg/8  �          A*ff�P�׿�G���8RCPW
�P��?W
=�!��{C%��                                    Bxg/'�  �          A*�\�j=q��\)�
=�fCD�3�j=q?����ff��Ch�                                    Bxg/6�  "          A+\)�k���z���\��CLJ=�k�?h��� (�� C&(�                                    Bxg/E*  �          A+��qG���p��{u�CL���qG�?Tz��   C'��                                    Bxg/S�  �          A%��j�H���  CL}q�j�H?G��aHC'�                                    Bxg/bv  "          A%��j�H��ff����CJ���j�H?h���ff8RC&�                                    Bxg/q  
�          A*ff�`  ��  �ff�CN���`  ?L��� Q���C'&f                                    Bxg/�  
�          A((��j=q�\��
�RCJ���j=q?xQ������C%#�                                    Bxg/�h  "          A(����  ��=q��\�)CFc���  ?����
=�\C$+�                                    Bxg/�  �          A&�R�l(��(����fC=aH�l(�?���W
C��                                    Bxg/��  
�          A&�R�p�׿z�H��\(�CB���p��?��H���k�C�\                                    Bxg/�Z  
�          A#���p�=L���\)�RC3J=��p�@ff�ff�r�C
=                                    Bxg/�   �          A&�H��p��8Q�����=C=33��p�?�\)��H�{ffC�                                    Bxg/צ  "          A-G�������ff(�CE����?�z���R�qC$}q                                    Bxg/�L  �          A%���e���R�{�CF�q�e?�Q��=q#�C!��                                    Bxg/��  
�          A)��z�ٙ���(�CJO\��z�?:�H��
\)C)��                                    Bxg0�  F          A)���p���{�ff�y  CG��p�?@  �  �~{C*�{                                    Bxg0>  T          A(�����޸R����w  CHk���?���
=�}C,�R                                    Bxg0 �  T          A(Q����R���p��{\)CPxR���R=��
���C2�                                    Bxg0/�  �          A%����Ϳ��R�z��r��CK5�����>�\)��
�}
=C0�\                                    Bxg0>0  �          A"{���H�z��
=�aCL����H�B�\�  �o�C6!H                                    Bxg0L�  
�          A!���33�`  ���
�<��CR����33�˅�   �SffCC@                                     Bxg0[|  
L          A{���
�o\)���
�:(�CU�����
��33��=q�S\)CF��                                    Bxg0j"  �          A\)�����o\)���H�>p�CWQ����ÿ�z���G��X�
CG�
                                    Bxg0x�  T          A
=�����;��(��i��CW5����ͿW
=��8RC?h�                                    Bxg0�n  T          A+��b�\��G�� (���CG���b�\?���� Q�C!B�                                    Bxg0�  
Z          A0z��\@�
�,(��RB���\@��H�
=�B��3                                    Bxg0��  b          A9���   @^{�.{Q�B�ff�   @�����i�B��=                                    Bxg0�`  
�          A:�H��@�\�3\)�)C���@�p��%G��zG�B♚                                    Bxg0�  
�          A;\)�>�R@��1G�(�CǮ�>�R@����#��t�RB��
                                    Bxg0Ь  �          A<  �G�@�1G�33C33�G�@��#33�p�B�                                    Bxg0�R  �          A9p��j�H@�R�,(��=C� �j�H@����R�kz�B��                                     Bxg0��  �          A.=q�w���\)� (�k�CG� �w�?��� ��z�C$��                                    Bxg0��  �          Aff������\)�������Ci�)������=q�Ϯ�1�\Cb��                                    Bxg1D  T          A�������Q������Q�Cm�f�������R�����Ch�f                                    Bxg1�  �          A!���p���(���Q�����Cn��p��ָR��Q��
�Ci�R                                    Bxg1(�  
�          A!��������R�s33����CnT{����ڏ\���\�=qCjJ=                                    Bxg176  
�          A�����   �Z=q���Cn�����
=��ff���Ck#�                                    Bxg1E�  �          A�\������z��L�����RCmY����������
=��
=Ci�\                                    Bxg1T�  "          A����R��\�K���z�Cj5����R�Ӆ��(�����Cfp�                                    Bxg1c(  �          A"=q���\� ���tz����Co\���\������\)Ck!H                                    Bxg1q�  �          A (���  ��{�tz���\)Co{��  ��=q��=q�ffCk�                                    Bxg1�t  
�          A������p��k���
=Cp�H������  ��
=�  Cl�R                                    Bxg1�  �          A ����  ������������Cn����  �׮��Q���
Cj�=                                    Bxg1��  �          A!���G���z����
��z�Cn� ��G��ָR��33�p�Cj}q                                    Bxg1�f  �          A#���{�����R�ۅCm  ��{�ə��˅�Cg��                                    Bxg1�  �          A%���33�����p���G�ClY���33���
�ʏ\�33CgJ=                                    Bxg1ɲ  �          A%����
������G���  Cl�f���
��Q���\)��RCg�{                                    Bxg1�X  �          A$Q����
���H������\)Cl����
���������HCf��                                    Bxg1��  �          A'�
��ff������=q���Cj�f��ff���R��z��$33Cd�                                    Bxg1��  �          A)�������Q���ff���Ck�����G���G��&��Ce�=                                    Bxg2J  T          A)�����  ��
=�z�Cg�������  ���*\)C`�                                    Bxg2�  �          A)����(����
��=q��RCe�)��(���G����2
=C]ٚ                                    Bxg2!�  �          A*{������  ���R�p�Cf)������ff���/{C^��                                    Bxg20<  �          A%���z���{��(��
{C`���z�����\�,G�CW��                                    Bxg2>�  �          A)�����ʏ\���
��\Caٚ������=q����*�CZ�                                    Bxg2M�  �          A-���\)�߮����� �\Ces3��\)��\)��
=�&�RC^��                                    Bxg2\.  T          A.�R��Q������=q��G�Cf����Q���  ���
���C`��                                    Bxg2j�  "          A1��Å��G����\��Q�Cg��Å��(�����p�Ca)                                    Bxg2yz  T          A2�\��33������  ��(�Cdc���33�����\)�!p�C]�                                    Bxg2�   0          A7��ҏ\��{���H��Cb�\�ҏ\��=q��  �)�CZ�q                                    Bxg2��  
~          A8Q���33�����ə���\Cb�\��33��p���\)�(�C[ff                                    Bxg2�l  "          A8z���Q�������G��z�C]�R��Q��������7(�CTQ�                                    Bxg2�  "          A;33��  ��{�����  Ca�{��  ��G����+{CZ)                                    Bxg2¸  "          A=����G���z����,��C\�=��G��j=q��H�K�CQG�                                    Bxg2�^  T          A=p���z�������p��(
=C\�q��z��u��(��Fz�CR                                    Bxg2�  �          A>�R���
�˅��  �
=C_�{���
��Q���
�=��CV@                                     Bxg2�  "          A@��������
=����#33C_�������������CQ�CT�H                                    Bxg2�P  �          AA����  ���\�=q�)��C\�{��  �w
=��
�GCQ                                    Bxg3�  
�          ABff������=q��\�)33CZc������fff�33�EffCO!H                                    Bxg3�  
�          AC�
���
��33�G��8��C]L����
�^�R��Vz�CP��                                    Bxg3)B  
�          ADQ���p���p�����*Q�CV����p��L(��(��C��CKE                                    Bxg37�  "          AD����z���ff����0�CZW
��z��Z=q����L�CNT{                                    Bxg3F�  
�          AF{���H����(��-z�CZ����H�a��z��I(�CNz�                                    Bxg3U4  �          AG
=������� (�� �RCZ��������Q����=G�CPs3                                    Bxg3c�  �          AG33��\��������-�CW����\�O\)�Q��G�\CK�{                                    Bxg3r�  �          AG�
�����z��2��CSǮ���'��p��H��CG#�                                    Bxg3�&  �          AIG������33�Q��$G�CQ�����*=q�G��8�RCEǮ                                    Bxg3��  
�          AI�
�\�����33��CP�R�
�\�9���33�.z�CF�                                     Bxg3�r  
�          AI�
=������\)CQ�R�
=�\���=q�!p�CI{                                    Bxg3�  T          AJ�\��������=q��
CM�����6ff��z��G�CD�                                    Bxg3��  �          AL  ��������Q��p�CL�{���!��Q��!��CC)                                    Bxg3�d  �          AMG��G���z���z���CM���G��#33�
�\�*33CC��                                    Bxg3�
  T          AO
=�$Q�������z����CK�f�$Q��B�\��\)��CD�                                     Bxg3�  �          AN�\�����
��{��CN�
���O\)���\��\CFff                                    Bxg3�V  "          AN�\�%���=q���R�ɮCOE�%���  �Ϯ��G�CI.                                    Bxg4�  �          AP����H�\)��R��CI�f��H�G��ff���C@޸                                    Bxg4�  �          AP���*ff�w
=���
����CG�f�*ff�Q������\C@�{                                    Bxg4"H  �          AP�����%��z��0Q�CD������{�8�C7��                                    Bxg40�  T          AQ�G���{��H�9
=C?���G�>k���=��C2�                                     Bxg4?�  T          AQ���ff�:�H�#��Mp�C9��ff?�G��"ff�K{C)�                                    Bxg4N:  �          AQ����33��G��'��T��C4�{��33@
=q�$(��N33C$��                                    Bxg4\�  �          AQG����>��H�*�R�^��C0&f���@1��$���SC�                                    Bxg4k�  �          AN�H����?�Q��8���}��C"�3����@�z��.ff�f��C)                                    Bxg4z,  T          AN�R���?xQ��2{�o  C+\)���@Vff�*ff�_G�Cn                                    Bxg4��  T          AM�����H?���4���v�C$J=���H@�G��*�R�ap�CaH                                    Bxg4�x  T          AL����Q�?�z��3��t��C&�\��Q�@r�\�*�\�aQ�C�                                    Bxg4�  �          AIG���z�>�G��*�\�h{C0=q��z�@-p��%��\�RC�                                    Bxg4��  �          AJ�\��  ?z��/��p��C.����  @:�H�)p��c��C��                                    Bxg4�j  "          AH����
=�   �)��i{CE����
=>��R�,  �o��C1#�                                    Bxg4�  "          AO�
���H���0(��k�HCD�)���H>��2�R�qffC/��                                    Bxg4�  �          AS�
��
=?���<���{��C!�=��
=@���1��dp�C��                                    Bxg4�\  �          AU����@
=�>�H�|��C�����@���3\)�c�RC�)                                    Bxg4�  
�          AV{��G�?h���C\)� C*B���G�@_\)�;��vp�C��                                    Bxg5�  �          AT����ff>aG��C��)C1�=��ff@4z��>=q�~=qC��                                    Bxg5N  "          AV�\��(�@��B�RC�q��(�@��R�6�H�jz�C	k�                                    Bxg5)�  "          AV�\�˅� ���6�R�l\)CI���˅���;33�u�
C4E                                    Bxg58�  �          AW���녿   �;��r�HC8aH���@   �9��mffC#{                                    Bxg5G@  T          AW33��(�?���9��o��C*B���(�@dz��1��_��C�                                    Bxg5U�  �          AX�����?���<  �q�C(�f���@o\)�3\)�`33CG�                                    Bxg5d�  T          A[33��(�?s33�?
=�r��C+�
��(�@\���7��c��C��                                    Bxg5s2  T          A\z���z�?�
=�A��vp�C'n��z�@|���8���cCG�                                    Bxg5��  �          A\z���ff?�\�@���s�
C$����ff@�Q��6ff�_p�C��                                    Bxg5�~  T          A\����ff@�\�A�v�C!Ǯ��ff@�G��6�\�`�C                                    Bxg5�$  T          A\  ��33@N�R�A�w��C
=��33@�ff�2�\�Y�C}q                                    Bxg5��  T          Ab{����@Q��F�R�{{C ������@�{�;33�cp�C!H                                    Bxg5�p  �          Af{���@(��O\)ǮCQ����@���B�\�h�C	u�                                    Bxg5�  T          Abff���@���K�C����@�  �?
=�g(�C
G�                                    Bxg5ټ  �          Ad(����R@,(��N=q
=C����R@��\�@���h�CY�                                    Bxg5�b  
�          Ad���7�@�{�PQ���B�\)�7�@��;33�]�\Bݞ�                                    Bxg5�  
�          A`z�u@�33�H��#�B�.�uA	G��/�
�P
=B��
                                    Bxg6�  �          A]�?�Q�@Ǯ�C��z=qB�z�?�Q�A
ff�*�\�J��B�ff                                    Bxg6T  �          A\  ?��@�  �?��yz�B���?��A��'33�J��B��
                                    Bxg6"�  �          A_33��{@����R�\
=B�\��{@�p��?��l�\B�{                                    Bxg61�  T          A`(���z�@�ff�N�\.B�8R��z�@����8���_�B�                                    Bxg6@F            Ac�
?}p�@\�M��B�z�?}p�A	��4���R
=B��H                                    Bxg6N�  �          Afff?��\@�(��R�R�B��{?��\A�
�;��Z33B��                                     Bxg6]�  
�          Ag�?��@����T  �B�#�?��A�\�<Q��Y��B��                                    Bxg6l8  T          AhQ�?��H@����P(��=qB�#�?��HAp��6�H�P
=B�aH                                    Bxg6z�  �          Ajff?�@���R�HW
B��3?�A(��9��R
=B�k�                                    Bxg6��  �          Aj{@@�G��O�
�z�RB�G�@Ap��6�\�M�B�k�                                    Bxg6�*  T          Ajff@\��@�  �Ep��fB(�@\��Aff�)��:��B��q                                    Bxg6��  �          Ab�R@�R@�(��=���cB�\)@�RA�R�!��5��B�                                      Bxg6�v  �          Ab�H@��@�G��?��fffB�G�@��A�#\)�8=qB��H                                    Bxg6�  �          A^�R@=q@�Q��:=q�bB�33@=qA  �=q�5
=B�z�                                    Bxg6��  b          Ac�
@:�H@�=q�=G��a\)B��@:�HA���!��4p�B�.                                    Bxg6�h            A`Q�@��@���:=q�c(�B�\)@��A���{�5�B��q                                    Bxg6�  �          A`Q�?��@����9p��aG�B���?��A��Q��2(�B�ff                                    Bxg6��  "          A\��?��\@���7\)�`��B��?��\A  ��\�2  B�.                                    Bxg7Z  �          A`��?!G�A�7
=�Z\)B�\?!G�A$���z��+{B��                                    Bxg7   �          A\(�>��@���1p��XB�(�>��A!���\)�)p�B�aH                                    Bxg7*�  �          AZ�R?G�@�{�4Q��^Q�B�p�?G�AG��\)�/(�B���                                    Bxg79L  �          AZ{?+�@�
=�0(��W��B�k�?+�A ���=q�(��B�ff                                    Bxg7G�  �          A[�?�Q�A�
�-��R33B���?�Q�A$���33�#Q�B��)                                    Bxg7V�  T          A\��@�HA ���.ff�Q�B��@�HA!�z��#�HB�\                                    Bxg7e>  
�          A_\)?��
@���7��]�B�aH?��
A�
�=q�/  B�ff                                    Bxg7s�  
�          A]�?�  A(��0  �R��B�p�?�  A%G��G��${B��                                     Bxg7��  
�          A_33@1G�A33�/33�N�RB���@1G�A$(�����!��B�Ǯ                                    Bxg7�0  
�          A^=q@5Aff�.{�N=qB���@5A#33���!=qB��                                    Bxg7��  "          A^�R@@���6{�\�
B��{@A�����/ffB��                                    Bxg7�|  T          A^�R@ff@�Q��7��^(�B�B�@ffA
=�33�0��B��q                                    Bxg7�"  "          A_\)@7�@�(��2=q�T{B���@7�A�
����'\)B��                                    Bxg7��  �          A]@I��@�z��4���ZffB���@I��Az�����.ffB�p�                                    Bxg7�n  
�          A^�\@E@�{�5G��Z\)B��f@EAp��G��.=qB�L�                                    Bxg7�  "          A]p�@5�@���0  �S(�B�W
@5�A�H��H�&z�B�(�                                    Bxg7��  b          A^ff@`  @�\�4(��XB�
=@`  A�����-z�B�z�                                    Bxg8`  d          A^�R@���@�
=�9�b�RBg��@���A
=� ���9p�B�
=                                    Bxg8  �          A^{@�33@�
=�=�k(�BS{@�33A (��'33�C�Bv
=                                    Bxg8#�  
�          A\z�@��H@�{�:�\�g�BL�@��H@�{�$Q��@�Bo�                                    Bxg82R  �          A`Q�@�  @��
�7��[��BEz�@�  A��   �5��Bf��                                    Bxg8@�            Ac�@���@ᙚ�*�\�D  BF  @���A���(���\Ba��                                    Bxg8O�  �          Ah(�@z�H@�z��>�H�jQ�Be@z�HA�H�'
=�A33B��                                    Bxg8^D  �          Ab�\?��H@�{�Tz�\)B�aH?��H@����@���iffB�L�                                    Bxg8l�  �          Ab�\?�(�@�G��W�B�B�ff?�(�@��F=q�u
=B�Q�                                    Bxg8{�  �          Ad  @33@�33�UG���B��H@33@�ff�A��iG�B�.                                    Bxg8�6  T          A`  @z�@�Q��P  z�B���@z�@�G��<Q��d�HB�G�                                    Bxg8��  "          AZ=q@�@z�H�M���Bx=q@�@�{�<���pB�8R                                    Bxg8��  �          AXz�?O\)@s33�N�H�=B�\?O\)@��H�>=q�w
=B��                                     Bxg8�(  h          AW\)>��
@�=q�H���)B�z�>��
@�  �4���e(�B�                                      Bxg8��  
Z          AUp�?n{@�33�?\)�~��B��)?n{A=q�(z��Pz�B��                                    Bxg8�t  T          AV�\?
=q@�{�>{�y��B���?
=qA\)�&=q�K(�B���                                    Bxg8�  |          AU��>�@�=q�;�
�v��B���>�A���#��H�B���                                    Bxg8��  r          ATQ�?}p�@�\)�:=q�v�B�  ?}p�A\)�"=q�H  B��                                    Bxg8�f  J          AS�
?
=@���:ff�y��B��f?
=A���#
=�K33B��)                                    Bxg9  
Z          AL  �O\)@�p��'��]�B��H�O\)A=q����/\)B�#�                                    Bxg9�  
�          AK�
���@�=q�(���`ffB�����A����R�2
=B��\                                    Bxg9+X  
�          AK�<��
@�(��-�jp�B��3<��
A
�H����;��B�Ǯ                                    Bxg99�  
�          AL���n�R@�Q��{�@�HB�k��n�RA33��\)��B�B�                                    Bxg9H�  T          AS
=��A���������HCff��Az��.{�>�\C�                                    Bxg9WJ  T          AQ���
�HA��������G�CE�
�HA33�_\)�v�\C\)                                    Bxg9e�  	�          ARff�33@�ff�����Џ\C	�
�33A33������C33                                    Bxg9t�  E          AQ��
=A
=��G�����C��
=Ap��a��z�RC�3                                    Bxg9�<  �          AQ���	��A���p���
=C�)�	��A��4z��F�HC5�                                    Bxg9��  �          AP���
=A33��(����C#��
=A(��A��V�RC�)                                    Bxg9��  �          AO\)��HA33���\���C���HA
=�.�R�B�RC                                    Bxg9�.  "          ANff�
ffA
=q������\C��
ffA�����#33C�3                                    Bxg9��  T          ANff��A����  ����C����A�������C�=                                    Bxg9�z  �          AJ�R��A	���=q��{C����AQ��  �$  C�                                    Bxg9�   
�          AG���\A
=�j=q����C5���\A  ����Cn                                    Bxg9��  �          AG���A�R�.�R�K
=C���A�׿aG���=qC��                                    Bxg9�l  �          AK�
���A��vff����Cp����A����RCs3                                    Bxg:  "          AK33��
Az��}p���C�=��
A�\�{�!�Cp�                                    Bxg:�  
�          AJ�\��(�A=q��
=��33C(���(�A�R���R��p�C                                    Bxg:$^  
�          AD�����A�
��������Cs3���A(����
��Q�B�aH                                    Bxg:3  
�          AD  ��ffAG���ff�ݮC�{��ffA���z=q���RB�z�                                    Bxg:A�  
�          AC33��(�A	���\)����C����(�A���*�H�J�\C k�                                    Bxg:PP  "          AA���p�A  �������
C���p�A
=�{�=G�C ��                                    Bxg:^�  "          AA���{A�\��p���C� ��{A{�(Q��I��C0�                                    Bxg:m�  
�          AA�ȣ�A�
��33��  B�G��ȣ�A
=�n{���B���                                    Bxg:|B  E          ABff��G�A\)��ff�߅B��R��G�A�R�u�����B�{                                    Bxg:��  
�          AB�H��p�A���{��{B���p�A=q�aG���Q�B���                                    Bxg:��  
�          AD����ffAp������(�B����ffAp��~{����B�p�                                    Bxg:�4  �          AD  ��ffA����\��p�B�\��ffA��z=q���RB��                                    Bxg:��  �          AB�H���A����
���
B��{���A��n{��(�B�8R                                    Bxg:ŀ  "          AB�\��G�A��{����B��R��G�Az��c33���B��                                     Bxg:�&  
�          AA����Az���(��݅B�\���A�
�o\)��  B���                                    Bxg:��  
}          A@������A�
��p���=qB�p�����A���S�
��
B�\)                                    Bxg:�r  
�          A@Q���Q�A�
��33��Q�B�8R��Q�A�H����8��B�u�                                    Bxg;   �          A>�H����A�H�w�����B�aH����Az��   �{B���                                    Bxg;�  �          A>�R���A����z����\C 0����A
=���0��B�k�                                    Bxg;d  �          A?
=��\A(��}p����HC ���\A{�
=�"�RB�\                                    Bxg;,
  "          A=���RA  �`�����C����RAz��
=�B��q                                    Bxg;:�  �          A?33��RA	��hQ����C���RA녿�=q�G�CE                                    Bxg;IV  "          A?\)���RA��O\)�{33C�q���RA�Ϳ��R���C!H                                    Bxg;W�  T          A>�\��p�AQ��QG��33C��p�A(����
����C)                                    Bxg;f�  �          A>{���
A�<���f�\CJ=���
AzῙ����=qC�H                                    Bxg;uH  "          A7�
��@�Q�@A�Ax(�C�)��@���@�A�G�C�3                                    Bxg;��  
�          A8���
=@�33@^{A�z�CW
�
=@���@���A�p�C&f                                    Bxg;��  
�          A733��R@�33@�\)A�(�C����R@L��@�  A�C �                                    Bxg;�:  T          A6�\�z�@���@�Q�A�  C=q�z�@Y��@\B ��Cff                                    Bxg;��  
�          A4z��(�@�(�@�z�A�=qC(��(�@\(�@ƸRB�C��                                    Bxg;��  �          A7��
=q@�=q@�G�A��HC�)�
=q@b�\@���B��C�3                                    Bxg;�,  T          A:�H�33@�@<��Aj=qC�
�33@�ff@�z�A�(�C�                                    Bxg;��  
�          A=��G�@�@QG�A�C.�G�@�z�@�ffA�Q�C��                                    Bxg;�x  "          A>�\�z�@��R@�p�AÙ�C�
�z�@��\@���A�C�{                                    Bxg;�  
�          A?33�p�@ƸR@Z�HA�33C\�p�@�z�@�33A�C��                                    Bxg<�  
�          A?��#
=@��@+�AO�
Cc��#
=@���@p  A���CT{                                    Bxg<j  
�          A>�\�'
=@�p�?#�
@E�C}q�'
=@�p�?��
A	p�C��                                    Bxg<%  �          A>ff��
@�@
�HA)p�CO\��
@��@U�A���C�                                    Bxg<3�  
�          A>�R�=q@�\)@���A��C�\�=q@�(�@���A��C��                                    Bxg<B\  �          A>{�p�@��@5A]�C���p�@��@��A���C�3                                    Bxg<Q  "          A<����
@��@AG�Am��Cff��
@��@��A���C��                                    Bxg<_�  "          A=G��ff@�=q@U�A��HC
�ff@�Q�@�G�A�
=C��                                    Bxg<nN  �          A>�\��R@��H@l��A��C�f��R@��R@��A�
=C�)                                    Bxg<|�  
�          A:ff���@��
@5Ab{C&f���@�p�@{�A��CO\                                    Bxg<��  
�          A:�R���@�@2�\A]G�C޸���@��@x��A��
C��                                    Bxg<�@  T          A:�H�@\@%�AL��C��@�@qG�A��HC�\                                    Bxg<��  
�          A;���\@�
=@#�
AJ{CxR��\@��@u�A��RCG�                                    Bxg<��  T          A;\)�
=@˅@-p�AVffC��
=@�p�@}p�A�{C��                                    Bxg<�2  "          A<  �
=@У�@S�
A���C���
=@��R@�=qA��C.                                    Bxg<��  "          A<(���@�  @g�A�p�C����@��
@�(�A�C\)                                    Bxg<�~  T          A=��33@�Q�@|(�A��C��33@��\@��HA��C��                                    Bxg<�$  �          A>=q��H@�p�@���A���C�f��H@�{@���A��HCY�                                    Bxg= �  T          A=G���\@��\@���A�z�C33��\@��@�Q�A�G�C��                                    Bxg=p  T          A=���z�@�  @���A��C��z�@�  @�A��C�3                                    Bxg=  "          A=��Q�@��H@��A�\)C�3�Q�@��H@���A�ffC:�                                    Bxg=,�  "          A=���p�@�=q@���A��RC���p�@��
@�AυCL�                                    Bxg=;b  "          A=��=q@��H@�\)A�p�C��=q@�33@��HA�z�C��                                    Bxg=J  	�          A<�����@�\)@uA�(�C����@�=q@�{A�(�C=q                                    Bxg=X�  �          A<Q���H@��@\)A�Q�CY���H@��
@�33A�G�C�3                                    Bxg=gT  "          A<(����@�z�@�33A�G�C�f���@�p�@�\)A�33C!H                                    Bxg=u�  �          A<Q���H@���@�Q�A�p�C� ��H@��\@��
A�=qC��                                    Bxg=��  "          A;���@�
=@p  A�33C}q��@�=q@���AŮC�
                                    Bxg=�F  
�          A:ff�Q�@�=q@G
=Ax(�C�R�Q�@�G�@�33A��C&f                                    Bxg=��  T          A;��
=@�@g�A��C�
�
=@��@���A�{C��                                    Bxg=��  �          A:�R�Q�@�(�@aG�A�z�C���Q�@���@��RA��RCaH                                    Bxg=�8  
�          A:�\���@���@fffA�  C����@�z�@���A�\)C�                                    Bxg=��  
Z          A;33���@��@e�A�ffC�����@��
@�{A�
=C�3                                    Bxg=܄            A:{�{@θR@J�HA}p�C�R�{@��@�{A��
C5�                                    Bxg=�*  
�          A<  �ff@�G�@L��A|��C=q�ff@�\)@�{A�p�C�                                     Bxg=��  �          A<����@�p�@l(�A��C�f��@�Q�@�z�AîC��                                    Bxg>v  T          A=p��@�G�@�Q�A��\C(��@�=q@�A�C��                                    Bxg>  T          A<���ff@�\)@z�HA��HC�=�ff@���@��\AˮC�
                                    Bxg>%�  T          A<  �ff@�ff@q�A�=qC���ff@���@�{A���Cٚ                                    Bxg>4h  T          A<(��33@��@o\)A�z�C���33@�  @�z�A���C!H                                    Bxg>C  �          A<(��(�@�\)@w
=A��C��(�@���@�\)A�ffC:�                                    Bxg>Q�  
�          A9p����@���@s�
A��C����@��@��\AŅC}q                                    Bxg>`Z  �          A9p���R@\@\��A�p�Cz���R@�
=@���A�Q�CaH                                    Bxg>o   �          A;33�@�p�@���A�C��@w
=@�ffA�  C!H                                    Bxg>}�  T          A:�R��H@�{@�  A���C����H@{�@�ffA�p�C��                                    Bxg>�L  �          A9p��Q�@���@g�A��C�f�Q�@�(�@�ffA�p�C�                                    Bxg>��  �          A9G���@�ff@(��AS33C&f��@�Q�@uA���C:�                                    Bxg>��  
Z          A8����@�{@%AP  C����@�Q�@uA���C�\                                    Bxg>�>  T          A8���@�33@Q�A>ffCٚ�@�ff@j�HA�  C��                                    Bxg>��  �          A9���
@��@A:�HC�H��
@��@j�HA�(�C^�                                    Bxg>Պ  "          A9p��=q@�R?�@:�HC
T{�=q@�p�@��A/�Cs3                                    Bxg>�0  
�          A9���33A
ff�z�H��z�CO\��33A
�\?^�R@��CE                                    Bxg>��  "          A9����ffA	G��������RC���ffA
=q?#�
@K�C��                                    Bxg?|  
�          A:�R��
=A
ff� ���\)C����
=A{����RC{                                    Bxg?"  �          A:�H�׮A�5��`  B����׮A(��Tz����B���                                    Bxg?�  �          A8����{A��6ff�e�B�p���{A(��W
=��ffB�#�                                    Bxg?-n  "          A;�
��HA{>�p�?�CJ=��HA@	��A(��C:�                                    Bxg?<  "          A<Q��  @�?��A Q�C���  @���@R�\A�=qC
�3                                    Bxg?J�  �          A<���G�A
=?�33@�33C� �G�A   @:�HAf�\CG�                                    Bxg?Y`  
�          A;\)��Q�A{��R�0Q�CE��Q�Aff��\)���C k�                                    Bxg?h  6          A<(���=qAp���ff��=qB��)��=qA33?
=q@)��B�B�                                    Bxg?v�  �          A<z���G�A�׾�녿�(�B�p���G�Aff?ٙ�A��B�8R                                    Bxg?�R  T          A<���أ�A=q>L��?z�HB�.�أ�A�@G�A1G�B��3                                    Bxg?��  
�          A:�R��ffA33?z�@5�CǮ��ffA	@�RAD��Cٚ                                    Bxg?��  T          A:=q��
=A	�?�  @��C
=��
=A ��@S�
A�(�C�
                                    Bxg?�D  �          A9���
=A�?��@˅B�(���
=A�@O\)A���C �)                                    Bxg?��  T          A9��A{?L��@|��B��
��A�
@0  A[\)C!H                                    Bxg?ΐ  T          A9�����
Ap�?��@���C {���
A
ff@>�RAn{Cu�                                    Bxg?�6  �          A9G���z�A��?.{@W�C Q���z�A
�H@'�AQCu�                                    Bxg?��  T          A9G��Q�@�ff@�\A&{C8R�Q�@��@UA��HC��                                    Bxg?��  �          A:�R�#33@���@"�\AIG�C���#33@��@dz�A���C�
                                    Bxg@	(  �          A9��#�@�{@�HA@Q�C5��#�@���@[�A��C33                                    Bxg@�  �          A:ff�%G�@�G�@8��Af=qCE�%G�@s33@s�
A���CǮ                                    Bxg@&t  "          A:=q�#\)@��@@��Ap  CxR�#\)@x��@|��A���C!H                                    Bxg@5  �          A<���"�R@��H@(Q�AN=qCL��"�R@�z�@n�RA��C}q                                    Bxg@C�  �          A;���R@�33@�A+\)Cu���R@��@Y��A�
=C.                                    Bxg@Rf  �          A<  �   @��H?�p�A
=C�R�   @���@Mp�A}p�C=q                                    Bxg@a  �          A:�H���@�33?��H@�p�C=q���@�(�@0��AZ�\C=q                                    Bxg@o�  4          A:�H�33@�Q�?�A�HC�3�33@�
=@AG�Aq�CQ�                                    Bxg@~X  �          A:ff�33@��R?��RAG�C+��33@�(�@L(�A~�\C��                                    Bxg@��  �          A:�R��R@�ff?\@�  C{��R@�
=@2�\A]��C(�                                    Bxg@��  �          A:ff��R@��@ ��A�C���R@��@N�RA���C��                                    Bxg@�J  �          A:�H�{@���?���A{C#��{@��\@Mp�A33C��                                    Bxg@��  �          A:ff���@�{@�RADz�C
=���@��@q�A��C�                                    Bxg@ǖ  �          A:=q�G�@��?�\)A Q�C8R�G�@�z�@@  Ao33Ck�                                    Bxg@�<  �          A9����@�  ?��R@�z�C�=��@�Q�@8��Ag�C�)                                    Bxg@��  �          A9p��p�@׮?��\@�G�C(��p�@�G�@.�RAZffC                                    Bxg@�  �          A9�\)@�{?�G�@�
=C  �\)@Ϯ@1G�A\��C�{                                    BxgA.  �          A9��G�@θR?�
=@�33C  �G�@�G�@%AN=qCǮ                                    BxgA�  �          A9��
=@Ӆ?�\)@���C�q�
=@�{@$z�AL��C�R                                    BxgAz  �          A9��(�@�Q�?z�H@�z�C�
�(�@�(�@=qA@��C0�                                    BxgA.   �          A8z��(�@�
=?=p�@j=qC�(�@�z�@
=qA-G�C#�                                    BxgA<�  �          A8���!��@��>�ff@��C.�!��@�=q?�Q�A�\C@                                     BxgAKl  �          A8(���@��H?�R@Dz�C����@�G�?�(�A��C8R                                    BxgAZ  �          A8  ��H@��
?G�@xQ�C0���H@У�@33A8��C��                                    BxgAh�  �          A8(��{@陚?
=q@+�C�\�{@߮@
�HA-C�                                    BxgAw^  �          A8Q��@�  ?���@�  C�3�@�G�@3�
AaC�                                    BxgA�  �          A6ff�陚A�@�A$Q�C���陚@���@x��A�{C{                                    BxgA��  �          A6{�p�@�\)@G�A#�CO\�p�@�=q@mp�A�z�C
�
                                    BxgA�P  �          A5G��	p�@���?���@�33C� �	p�@��@4z�Af�RC�
                                    BxgA��  �          A5��Q�@�33?c�
@�  C  �Q�@ָR@{AJ{C�\                                    BxgA��  �          A5p��
=@���?J=q@�Q�C
@ �
=@���@�RAJ�\C��                                    BxgA�B  �          A4����\@�R?}p�@���C
c���\@�G�@*=qAZffC
=                                    BxgA��  �          A3��33@�G�?�(�@�
=C5��33@�=q@6ffAk
=C
                                    BxgA�  �          A3�
�G�@�(�?��H@�RC
u��G�@�33@FffA�ffC��                                    BxgA�4  �          A4z���
@�?�z�@�p�C	G���
@���@7�Al(�C{                                    BxgB	�  �          A4���  @�(�?��\@�p�C\�  @�ff@+�A[�C�                                     BxgB�  �          A4���(�@��?n{@�\)C!H�(�@��@!�AN�HC                                    BxgB'&  �          A4Q��\)@��?}p�@���C���\)@�z�@%AT��C�                                    BxgB5�  �          A4z����@�G�?W
=@���C����@���@   AMG�C�                                    BxgBDr  �          A4Q��  @�?�{A(�C	���  @ۅ@Q�A�(�C@                                     BxgBS  �          A4z��
=@�z�?��
@�\)C
�=�
=@޸R@-p�A^�\C��                                    BxgBa�  �          A4�����@�?�\)A�C
O\���@ۅ@R�\A���C��                                    BxgBpd  �          A5��ff@�p�?�33A�C� �ff@�\@XQ�A��C�                                    BxgB
  �          A4���
�\@�(�?�p�@�Q�C���
�\@���@7
=AjffCu�                                    BxgB��  �          A4����H@Ӆ@�A)p�C����H@�{@c�
A���C^�                                    BxgB�V  �          A4���\)@�Q�@3�
Af{C
� �\)@��@�A�=qC�                                    BxgB��  �          A4�����@�(�@N�RA�\)C
�����@�@��A�{C��                                    BxgB��  �          A4z��ff@�ff@ffA+�Cc��ff@�Q�@n{A�z�C8R                                    BxgB�H  �          A5G���@�@,(�A\(�C33��@�33@���A���C�f                                    BxgB��  �          A5���@��@J=qA��
Cu���@�
=@�{A�=qC}q                                    BxgB�  T          A4z��{@�\)@~{A���Cff�{@��@�{A�RCk�                                    BxgB�:  �          A3�� ��@ʏ\@�Q�A�Q�C��� ��@��H@��
A���C�                                    BxgC�  �          A1��  @�p�@   A&{CG���  @�\)@p  A�=qC	�R                                    BxgC�  T          A2�\��
=A�\������
C�H��
=A�R?z�H@�G�C��                                    BxgC ,  �          A1����p�AG��Tz����\C�3��p�A��?�@�=qC�
                                    BxgC.�  �          A1G�����AQ�>��R?У�C �q����A33@z�AC�
C\                                    BxgC=x  �          A1���\)A	���Ǯ�{C ���\)A�?�@0  B�G�                                    BxgCL  v          A/33��Q�A�R��������B�G���Q�A  ?E�@���B���                                    BxgCZ�  �          A/33�ə�Az���H�  B�G��ə�A
=>�(�@�B�W
                                    BxgCij  �          A/33���\A�\�ff�1�B����\A=q>��?L��B�ff                                    BxgCx  �          A1�����A���-p��c33B�aH����A�R�����B�p�                                    BxgC��  |          A1�����A���\)��z�B������A������
B��                                    BxgC�\  �          A0(���p�A��Tz���  B�z���p�A  �������RB�R                                    BxgC�  �          A/����@�z�8Q��r�\C�H���@��H?��@�
=C�{                                    BxgC��  �          A/33��\)@��(��9�CB���\)@�\)��\)����C(�                                    BxgC�N  �          A-���{@��
�33�.ffCh���{A=q��Q��Cz�                                    BxgC��  �          A-���p�A�\���#33CG���p�A{=�G�?z�C�                                    BxgCޚ  |          A0(���z�A��
=��  B�R��z�A�
��H�L��B�=                                    BxgC�@  �          A0�����A\)�k���z�B��
���Ap�������(�B�                                     BxgC��  T          A/
=���H@�\��p���C�R���H@�{?��R@��C�                                     BxgD
�  �          A0(���  @��R�W
=���
C(���  @�?���@�33CE                                    BxgD2  �          A0z��g�@�{��  ��
B�
=�g�A����33��33B�W
                                    BxgD'�  �          A0���(Q�@������5Q�B۔{�(Q�A(����\����BԊ=                                    BxgD6~  �          A.�H�A�@�33��\)�-  B��
�A�A\)��ff��p�B�aH                                    BxgDE$  �          A.�\���@��
�p��@
=B�k����A
ff�����	��B���                                    BxgDS�  �          A'�
����A�\�c�
���HB�G�����AzῸQ�����B�ff                                    BxgDbp  T          A'33��(�A�
�E����B����(�A  �s33���RB��=                                    BxgDq  �          A'���G�@�
=�tz����B�����G�A
�\��p��z�B�33                                    BxgD�  �          A*�\��G�@�  ������p�B�
=��G�A���K���33B�                                     BxgD�b  �          A)�����@��H��(���(�B�������A��AG�����B���                                    BxgD�  �          A(�����R@�33��(���
=B��f���RA
=q�E����B�8R                                    BxgD��  �          A'�
���@��
��(��	
=B������A�z=q��(�B�\)                                    BxgD�T  �          A&�H��  @�����
��z�B�� ��  A33�G��IG�B�8R                                    BxgD��  T          A(Q���ffAp��W���G�B�\)��ffA�R����z�B��f                                    BxgDנ  �          A$�����HA(�@'
=As�B�3���H@��H@�33AܸRB��q                                    BxgD�F  �          A�R���HA�H?�  A  B�B����H@�G�@`��A�z�B���                                    BxgD��  �          A ����{@���@S33A�\)B��H��{@�ff@���A���C�                                    BxgE�  �          A!����
=@���@�p�B(�CQ���
=@��R@ڏ\B*�C�                                    BxgE8  �          A\)����@��
@�  A��B�W
����@�  @��B��C��                                    BxgE �  �          A(���=q@ۅ@�ffA�RB�����=q@�ff@���B ��C�R                                    BxgE/�  �          A&{��G�Aff@Q�A���B���G�@ᙚ@��A���B���                                    BxgE>*  �          A(Q�����A{@U�A��B��R����@��@�G�A�Ck�                                    BxgEL�  �          A'33����@�\)@s�
A�\)B��3����@׮@�
=B�
C}q                                    BxgE[v  �          A&�H��@�@�G�A�Q�B�����@�(�@�B
�CJ=                                    BxgEj  �          A'���{@�@���B =qB�Ǯ��{@���@�ffB.(�CE                                    BxgEx�  �          A(����p�@�  @���A�(�B�u���p�@�Q�@�(�B�HC�{                                    BxgE�h  �          A)���p�@���@�{A�B�
=��p�@�{@�  B��CxR                                    BxgE�  �          A)���\)@�p�@���A��B��)��\)@��@ÅB\)Cs3                                    BxgE��  �          A(�����@�@l��A�
=C�H���@��H@���A��C�)                                    BxgE�Z  �          A)G���(�@���@mp�A�=qC =q��(�@��@�33A�p�C{                                    BxgE�   �          A'\)���A   @X��A�p�B������@�33@��HA�(�CL�                                    BxgEЦ  �          A&ff��
=A ��@]p�A�z�B�W
��
=@�z�@�A��C xR                                    BxgE�L  �          A'����\@�33@�
=A���B��\���\@�
=@�\B*  B��q                                    BxgE��  �          A'����
@��@�33B=qB������
@�{@��HB1�RC��                                    BxgE��  �          A$z���33@�ff@��B�B�33��33@�
=@��HB7{C{                                    BxgF>  �          A#33��z�@�Q�@���B\)B����z�@�p�@�ffBE��C�R                                    BxgF�  �          A"=q���H@�\)@��\BJ�\C�����H@!G�A��Bq=qC��                                    BxgF(�  �          A Q���ff@�z�@�(�BQp�C����ff@
�HA(�Bv�C��                                    BxgF70  T          A&�\����@��H@�B3�
B�������@{�A33Bb�C��                                    BxgFE�  �          A!�����@��H@�p�B(�B�k����@�
=@�  BI�CE                                    BxgFT|  �          A!���s33@��@�G�B0�B�k��s33@�=qA�RBb  C�                                    BxgFc"  �          A �����@�Q�@θRB z�B��)���@�=q@�ffBQ��C��                                    BxgFq�  �          A$  ���R@�  @�G�A�Q�B�k����R@�z�@�33B+ffB�.                                    BxgF�n  �          A(�����\Az�?^�R@�Q�B��)���\A\)@XQ�A��B癚                                    BxgF�  �          A)p��vffAG�=�Q�>��HB޸R�vffA33@-p�AmB�G�                                    BxgF��  �          A&�\�x��A{?
=q@@  B���x��A=q@FffA��\B�{                                    BxgF�`  �          A(���l��Ap�?\(�@�{B�33�l��A  @^{A�\)Bߔ{                                    BxgF�  �          A,(��s33A ��>u?��
B�p��s33A�@;�A|��B�#�                                    BxgFɬ  �          A-G��\(�A#\)�(���_\)B�G��\(�A (�@
=A4Q�B���                                    BxgF�R  �          A,Q��`��A!p����H��\)B�aH�`��A ��?�ffAQ�Bڊ=                                    BxgF��  �          A+��UA �Ϳ�Q��G�BظR�UA"{?�=q@�G�B�p�                                    BxgF��  
�          A2=q��ffA�H<��
=�B����ffA��@,��Ad(�B�aH                                    BxgGD  �          A<����A�?���A�B� ��A��@�\)A��B�8R                                    BxgG�  �          A;���
=A(�@(Q�AN�HB��H��
=A
�H@�A�\)B�                                    BxgG!�  T          A<����ffA!�?��@��\B�33��ffA=q@r�\A�z�B��q                                    BxgG06  �          A;33����A'33?�p�@�G�B�{����A�@���A���B��                                    BxgG>�  �          A:�\��G�A(  @z�A7�B���G�A�@�33A�p�B�=q                                    BxgGM�  �          A6=q���A'�?��R@���B�
=���A�\@���A�p�B�W
                                    BxgG\(  �          A4  �z=qA'�?�ff@��
B���z=qA��@x��A��
Bߊ=                                    BxgGj�  �          A9p���  A&{?�33A�B�33��  Az�@�{A�
=B�                                    BxgGyt  �          A=����\A((�?�ffA��B�����\A��@�(�A��\B���                                    BxgG�  �          A?
=��  A(��@9��Aa�B���  Ap�@�ffA�Q�B���                                    BxgG��  �          A<(���  A ��@(Q�AO33B�\)��  A�R@��A��HB�=q                                    BxgG�f  �          A9p���(�A
=@#�
AL  B��H��(�Ap�@��RA��
B�                                    BxgG�  �          A8Q�����A
=@
=A;�
B�.����Aff@���A�G�B�q                                    BxgG²  �          A4  ��{A�@�A.=qB�8R��{A=q@�G�Aə�B�.                                    BxgG�X  �          A2�H��ffA�
@
=AB�RB��H��ffA
=@�\)Aә�B�L�                                    BxgG��  �          A.�\���RA�
@	��A5��B�(����RA(�@��RẠ�B�p�                                    BxgG�  �          A/\)����@��@�ffA�G�B�������@�z�@�
=B*C��                                    BxgG�J  �          A<(���(�A�@�{A�(�B�aH��(�@ə�@���B%�C8R                                    BxgH�  �          A?���
=A\)@��HA���B�L���
=@�p�AffB+�RC=q                                    BxgH�  �          A@(����A  @��A���B�#����@�=q@��\B"�RC�)                                    BxgH)<  �          A<����Q�A{@�A�G�B��=��Q�@�Q�@�=qB�HC��                                    BxgH7�  T          A>ff�ϮA{@��A�z�B��=�Ϯ@У�@�Q�B�
C�H                                    BxgHF�  �          A7�����A�@�33A��HB������@θR@�B#ffC�=                                    BxgHU.  �          A5���\)A�R@���A�p�B���\)@�  @�ffB*��C ٚ                                    BxgHc�  �          A6{��=qA{@�ffA�B�8R��=q@�\)@�B&�RC��                                    BxgHrz  �          A733��z�AG�@��HA�  B��\��z�@���@�  B  C�                                    BxgH�   �          A;33��ffA	G�@�(�A�z�B����ff@���@�(�BQ�C
                                    BxgH��  �          A=G���
=A�@�{A�ffB�.��
=@�p�@�Q�B(�C(�                                    BxgH�l  �          AA��(�A�
@P��Az{B�p���(�Aff@�p�A�ffB�p�                                    BxgH�  "          A:�\���A33@_\)A��B�� ���@�=q@�p�A���C�                                    BxgH��  �          A7
=����A�\@^{A��B�\)����@���@�z�A�(�C�{                                    BxgH�^  T          A5��=qA��@��A�{C ^���=q@أ�@�G�B�RC#�                                    BxgH�  �          A6=q���@�33@�=qA��RC�\���@��H@�p�Bz�C
�                                    BxgH�  �          A733��A
{@;�Am��C ��@��@��A��C{                                    BxgH�P  T          A=p���Q�A��@5A^ffB�33��Q�@�(�@��
A�C�f                                    BxgI�  T          A=G���33A�
@:�HAe�C Q���33@�  @�p�A�{C�                                    BxgI�  �          A<����  Aff@Dz�Aq��B�����  @�33@�33A�\)C��                                    BxgI"B  T          A7�
�ҏ\Ap�@FffA{\)B�G��ҏ\@�@���A�  C\                                    BxgI0�  �          A4Q���\)A��@_\)A��RC���\)@�p�@���A��C5�                                    BxgI?�  �          A.=q�ӅA z�@J�HA���Cu��Ӆ@׮@�z�A��Cn                                    BxgIN4  �          A333����A\)?Y��@��HB�\����AQ�@o\)A�=qB��                                    BxgI\�  �          A2�R���A=q?z�H@��B�L����A�R@q�A���B�33                                    BxgIk�  �          A2=q����A33?s33@���B������A  @l��A�\)B��                                    BxgIz&  �          A1��z�AG�?�\)@�G�B�\��z�A	G�@u�A��B���                                    BxgI��  �          A2�R��{AG�?Q�@��B���{A
�\@c33A�\)B��                                    BxgI�r  �          A2{��\)A
=�5�i��B�=q��\)A�@�
A*�\B�u�                                    BxgI�  �          A0�����
A�R�z��@  B�u����
A�H@�A6=qB��f                                    BxgI��  �          A/
=���Az�(���\��B�B����A��@{A;\)B�                                     BxgI�d  T          A\)����@�=q����"�\B�녿���@����S33���HB�=q                                    BxgI�
  �          A=q=��
@�z������6��B�Q�=��
@��
�tz��ᙚB��                                    BxgI�  �          A���@�����  �:Q�B�p���@�{������G�B�                                      BxgI�V  �          A  ���R@�Q���
=�BBә����R@�z����H��\)B�aH                                    BxgI��  �          A	��s33@������
�	33B�.�s33@��4z���=qB�Q�                                    BxgJ�  �          A
�\�_\)@�p���ff�(�B�B��_\)@���^�R����B�                                    BxgJH  �          A	�����@����H�HffB��)���@�ff�����ffB�8R                                    BxgJ)�  �          A
{�9��@������J�
B�u��9��@����{�Q�B�                                    BxgJ8�  �          A
=q�p�@��У��D�RB�.�p�@�p���(��(�B�k�                                    BxgJG:  �          AQ��  @�����ff�<��B�=q��  @�p����R��
=B˙�                                    BxgJU�  �          @�z��\(�@�p�=�\)?&ffB����\(�@�?��RAn{B�                                    BxgJd�  �          @�
=�Vff@/\)@�
=BN��C�q�Vff?n{@�G�Bp\)C$u�                                    BxgJs,  T          @��
�X��?��R@���Bs�CJ=�X�ÿ�R@���B|=qC>ff                                    BxgJ��  3          @����w
=@��@�ffBR��C:��w
=>�ff@�z�Bj�HC-k�                                    BxgJ�x            @����ff@8Q�@�
=B7�\C#���ff?�ff@��HBTQ�C&��                                    BxgJ�  �          A{�p  ?��
@�Bk
=C���p  �(�@ڏ\Bs�\C=0�                                    BxgJ��  �          @�p���(�@QG�@�(�BQ�C�R��(�?�
=@�p�B/ffC"�3                                    BxgJ�j  �          A���p�@1�@��BD��C�H��p�?B�\@��
B_z�C*:�                                    BxgJ�  �          A ���c33@�@��
Bl��CE�c33�L��@��B~33C7G�                                    BxgJٶ  �          A�
����?�  @��Bi��C�������B�\@�(�Bp
=C>�f                                    BxgJ�\  �          A	�����@+�@�ffB4�C+�����?0��@ϮBJ
=C,��                                    BxgJ�  �          A�
��ff@�33@#�
A�Q�C���ff@��@��HA�Q�C
�                                    BxgK�  �          Az���ff@�\)@:�HA�G�C0���ff@�  @�
=B(�C	u�                                    BxgKN  �          A\)��(�@У�@*=qA�(�C 0���(�@�33@�=qA��C                                    BxgK"�  �          AQ���(�@�33@��A�{B��f��(�@�\)@�\)A�=qCc�                                    BxgK1�  �          A
�\���@�  @1G�A��C�f���@��\@�{A���C�                                    BxgK@@  �          A�����H@���@u�A�  C�f���H@s33@���B�C��                                    BxgKN�  �          A(�����@�@�=qAݙ�C
\����@hQ�@�  B��C�{                                    BxgK]�  �          A�����@�Q�@O\)A�\)Cz�����@��@���B	Q�C
h�                                    BxgKl2  �          A�����@�G�@'
=A��C�3����@�(�@�
=A���C	�                                     BxgKz�  �          A���z�@��@H��A�\)C ���z�@�=q@���B�Ch�                                    BxgK�~  �          AQ���ff@�=q@1G�A�Q�Cs3��ff@��
@��A�G�C�R                                    BxgK�$  �          A  ��\)@���@u�A��C�3��\)@q�@�=qB�\C�q                                    BxgK��  �          A�R���@�G�@~�RA��CG����@A�@�
=Bp�C0�                                    BxgK�p  �          A����p�@��H@x��A˙�C���p�@7
=@�=qBp�C��                                    BxgK�  �          Aff��
=@�z�@tz�AŅC�R��
=@Y��@�{B
G�C(�                                    BxgKҼ  �          Aff��{@�(�@n{A�z�C���{@j=q@�B��C33                                    BxgK�b  �          A�H��@��@�  A��C}q��@o\)@�Q�B�C�                                    BxgK�  �          A�R��Q�@��@\��A�{C����Q�@���@��B=qC}q                                    BxgK��  �          A
=��=q@�p�@O\)A��\C@ ��=q@��@���A�=qC�                                    BxgLT  T          A��ٙ�@�{@VffA��C
=�ٙ�@���@���A��C��                                    BxgL�  �          A��{@��@W�A��C���{@w
=@�z�A�33C�                                    BxgL*�  �          A(��ڏ\@���@'�A��
Cc��ڏ\@��@�=qA���C��                                    BxgL9F  �          A��33@��@�Amp�C
�H��33@�\)@�33A�Q�C\)                                    BxgLG�  �          A���{@��@G
=A�p�C޸��{@��@�Q�B�C}q                                    BxgLV�  �          A
=q�ʏ\@�Q�@)��A��RCG��ʏ\@�33@�\)A�=qC{                                    BxgLe8  �          A33�У�@�
=@�A���CW
�У�@�(�@�Q�A�p�C��                                    BxgLs�  �          A
=�θR@�33@�Ar�HCff�θR@���@z=qA��
C^�                                    BxgL��  "          A
�R��G�@�  @z�Ax  C
�
��G�@�@�  A��C�)                                    BxgL�*  �          A
{��{@��@�RApQ�C
{��{@�Q�@|��A�  C�3                                    BxgL��  �          A	���{@���@��Ag�
C
:���{@�Q�@vffA��
C�                                    BxgL�v  �          A�
����@�33@
=Af�\C	�����@��\@vffA�C                                    BxgL�  �          A�����R@���@%A�
=C	!H���R@��@���A�RC�{                                    BxgL��  T          A
�\��
=@�@�RA���C
�f��
=@�G�@���A��Cff                                    BxgL�h  �          A����=q@�G�@�RA�G�C
��=q@���@�{A�(�C.                                    BxgL�  �          AQ���33@��\@G�A�  C
�3��33@�
=@��B �Cp�                                    BxgL��  �          A��33@�Q�@^{A�
=Cn��33@r�\@���B  C)                                    BxgMZ  �          Aff���@��\@FffA�(�C�f���@��R@�G�A�ffCO\                                    BxgM   �          A�H�Ӆ@��@3�
A�ffC
=�Ӆ@��R@��\A��HC�                                    BxgM#�  �          A(����@��@�
ANffC�=���@��@���A�ffC)                                    BxgM2L  �          Az���33@�=q@=p�A��RC
�)��33@�\)@�  A�=qC�)                                    BxgM@�  �          A�����@Å@Q�Al��C	u����@�{@��A��
Ck�                                    BxgMO�  �          A������@��@S33A�C	L�����@�{@�33B�CB�                                    BxgM^>  �          A����
@��
@dz�A�=qC
�\���
@��\@���Bp�CaH                                    BxgMl�  �          A  ��33@��H@j�HA��C����33@qG�@���B  C8R                                    BxgM{�  �          A����
@��H@C�
A�
=C&f���
@~{@�ffA�
=C��                                    BxgM�0  T          A����G�@��@�A_
=C
=��G�@w�@n�RA�z�C�                                    BxgM��  "          A���@��@ffAo�C�
��@mp�@vffAȏ\C:�                                    BxgM�|  "          A���ff@�ff@2�\A�ffC����ff@]p�@�\)A�  C{                                    BxgM�"  
�          A�
��p�@��\@
=Af{Cz���p�@}p�@}p�A�G�C�3                                    BxgM��  �          A������@��R@#33Av�RC�=����@q�@�33Aʣ�C#�                                    BxgM�n  �          A�����@���@#33Aw33C
=����@}p�@�p�AθRC�f                                    BxgM�  �          AQ���=q@��@��Ai��CO\��=q@��H@��A�(�C�H                                    BxgM�  �          Az���ff@�
=@33AFffC����ff@�@n�RA�=qC}q                                    BxgM�`  �          A����@���@	��AO�C����@�{@uA��C^�                                    BxgN  �          A�\� ��@�p�@�AJffC�=� ��@xQ�@l��A��CB�                                    BxgN�  �          A{��z�@���@   A@  Cٚ��z�@�(�@j�HA�G�CY�                                    BxgN+R  �          A����33@���@
=qAW
=CO\��33@�{@w�Aď\CL�                                    BxgN9�  �          A����(�@�p�@!�A|  C��(�@�ff@���A�{C��                                    BxgNH�  �          Az����@���?��A2=qC�)���@�{@^�RA�=qC�)                                    BxgNWD  �          A���=q@�Q�@�Ab�RCW
��=q@��
@~�RAʏ\C�H                                    BxgNe�  �          A\)���@�G�?��HAE�C�����@�Q�@l��A��C�f                                    BxgNt�  �          A\)��=q@���?���A6�RC+���=q@���@dz�A�ffC��                                    BxgN�6  �          A�H��{@��?���A7
=C���{@�p�@g
=A�p�Ch�                                    BxgN��  �          A���33@�Q�?���A6{Cc���33@���@c�
A��
C�=                                    BxgN��  �          A  ��(�@���?�A4(�Cu���(�@�G�@c�
A�
=C�{                                    BxgN�(  �          A(���33@���?���A8��C5���33@���@g
=A��C��                                    BxgN��  �          A���  @��?�\A0��CB���  @�@e�A���C�{                                    BxgN�t  �          A�H��R@��?��A<��Cc���R@��H@j=qA�=qC�R                                    BxgN�  �          A�
��G�@�z�?��HA*ffC����G�@�p�@aG�A�G�C�                                     BxgN��  �          A�
��  @�\)?�{A
=C!H��  @�z�@H��A��RC�H                                    BxgN�f  �          A=q���
@��?k�@�(�C�����
@�
=@333A��RC:�                                    BxgO  �          A  ���@�
=?��A Q�C����@�@?\)A��Cff                                    BxgO�  T          A���Q�@�  ?��
@�{C{��Q�@���@6ffA�{C��                                    BxgO$X  �          A���@�p�?(�@s�
C���@��\@!G�A~=qC��                                    BxgO2�  �          A�H��ff@�  ?���@��C����ff@��@<��A��
C�f                                    BxgOA�  �          A
=��@��?E�@��C�H��@�G�@%A���C                                      BxgOPJ  �          A�R���@��R?Q�@�(�C����@��\@#�
A��C�                                    BxgO^�  �          A
=��@��\?J=q@�CW
��@�@*�HA�G�CǮ                                    BxgOm�  �          A�H���@��
?8Q�@�Q�C
=���@��@'�A���C^�                                    BxgO|<  �          A�R��33@�
=>�z�?�=qCY���33@��@G�Ad��Cٚ                                    BxgO��  �          A�\��\)@��
=�Q�?
=C!H��\)@�ff@��AX  CG�                                    BxgO��  T          A���@�ff<�>#�
CL���@���@
=AUCT{                                    BxgO�.  �          Ap���(�@�(�>.{?��C�q��(�@�@p�Aa��C
=                                    BxgO��  �          A(���=q@��    ���
CQ���=q@�ff@	��A]�CY�                                    BxgO�z  �          A�
���@���>8Q�?�
=C����@��\@��Ac�CL�                                    BxgO�   �          A����=q@�(�>���?�\)C}q��=q@�(�@ffAp��C{                                    BxgO��  �          A����{@���=���?(��C5���{@��H@�RAdz�Cs3                                    BxgO�l  �          A���{@Ǯ���Y��C����{@�33@�AaG�C	�                                     BxgP   �          A(���p�@�z�>\@�B�\)��p�@Ϯ@@  A�{C�{                                    BxgP�  �          A33��=q@�G�?J=q@��B�����=q@�\)@[�A�33C 
=                                    BxgP^  �          A�����@�?�@Z=qB�G����@�\)@I��A�{B���                                    BxgP,  �          A	����@��ͼ��
����Cٚ���@�ff@�A�
C�3                                    BxgP:�  �          AQ���\)@�{�B�\����C����\)@�=q@Q�AnffCJ=                                    BxgPIP  �          AG����@˅?�
=A+�
B�#����@��
@l��A�\B�G�                                    BxgPW�  �          A�H����@���@��RB=qB������@HQ�@��HBL\)CY�                                    BxgPf�  �          Az��G�@�33@�(�B�B�W
�G�@��@���Bb�HB�                                      BxgPuB  �          A  ��@�\)@�z�B��B�{��@|(�@��
BdB�                                    BxgP��  �          A�R�(Q�@�\)@�
=A��HB�33�(Q�@�p�@�33BK��B�=                                    BxgP��  �          A
=�>�R@ٙ�@Y��A��B�W
�>�R@�Q�@��RB1��B�p�                                    BxgP�4  �          Aff�z�H@��H@tz�A�(�B��3�z�H@~{@��B7ffC��                                    BxgP��  �          A���  @w�@�ffB��C����  ?�@��B9�C!�                                    BxgP��  �          A�\��@i��@���A�CaH��?�@�(�BQ�C"��                                    BxgP�&  �          A����@�=q@G�A��RC����@W�@uA�33CJ=                                    BxgP��  �          AQ����
@�  ?��A7\)C�f���
@�  @Z�HA�p�C��                                    BxgP�r  �          AQ���{@�{?�G�A(��C����{@\)@Q�A��C8R                                    BxgP�  �          A��ə�@�Q�?�p�Az�C#��ə�@���@I��A�  C�                                    BxgQ�  �          A��(�@�G�?���A(�Cp���(�@�z�@Tz�A���C:�                                    BxgQd  �          A������@��
?�G�AD(�C������@���@qG�A�33C�                                    BxgQ%
  �          A{����@��������C{����@��>��@:�HC��                                    BxgQ3�  �          @�  ��G�@s�
��\)�$(�C�=��G�@�G�>L��?��HCG�                                    BxgQBV  �          @�(���  @��;�{�p�C����  @�{?���A!�C+�                                    BxgQP�  �          @������@���>W
=?\CB�����@��\@�Al(�C�                                    BxgQ_�  �          @���p�@w
=��\)��C����p�@e?�A%p�C�\                                    BxgQnH  �          @�\)��z�@�\)?��A@��B����z�@�{@h��A�  C)                                    BxgQ|�  �          @��
��=q@�zᾮ{�#�
C�H��=q@�33?�\AXz�C5�                                    BxgQ��  �          @���Ǯ@��H?J=q@�{C33�Ǯ@l(�@=qA��Cc�                                    BxgQ�:  �          @��R��Q�@�
=>�?}p�Cz���Q�@�=q?�{A_�C                                      BxgQ��  �          @��H����@��    <��
C�����@�G�?�A_�C�3                                    BxgQ��  T          @�R��@c33?Tz�@�{C{��@<��@	��A��C\)                                    BxgQ�,  �          @��W�@��H?�(�AX  B����W�@��
@UB ffCJ=                                    BxgQ��  �          @񙚿(��@�ff@VffA��
B�=q�(��@�33@�ffBE\)B�u�                                    BxgQ�x  �          @��
�L��@�@>{A��\B���L��@�\)@��HB:��B�B�                                    BxgQ�  �          @�{��  @Ӆ@�
A���B͏\��  @���@��BffB�                                    BxgR �  �          @�33�^�R@�
=��\��{B�Q�^�R@���@z�A�B��                                    BxgRj  �          @����{@C�
?�A��
Cu���{@	��@/\)A�=qC#�                                    BxgR  �          @�����(�@ ��>���@J=qC����(�@	��?�{A*�HC"�                                    BxgR,�  T          @�z����
@{?���AA�C"�3���
?�z�@ffA�\)C(�{                                    BxgR;\  �          @�\)�陚@�
?��RA1�C$E�陚?�ff@�A�33C)��                                    BxgRJ  �          @��H��@�H?�G�AO�C!����?\@$z�A�p�C(�                                    BxgRX�  �          @�ff���H@G�@
�HA~�RC$�)���H?}p�@2�\A�\)C,L�                                    BxgRgN  �          A   ��R?�G�@�A~=qC&���R?@  @,��A��RC.:�                                    BxgRu�  �          A Q���p�?�p�?���AQp�C)
��p�?!G�@  A���C/G�                                    BxgR��  �          AG����
@��?�
=A#�C#u����
?�G�@{A~�HC(�{                                    BxgR�@  
�          @�(���\)@���?���AXz�Cc���\)@N{@X��A�=qC�3                                    BxgR��  �          @������@��
?�z�AB{C�\����@�Q�@j=qAݮCY�                                    BxgR��  �          @�����@�\)?�  A1��C\��@`��@J�HA�(�Cn                                    BxgR�2  �          @��\����@�(�?��AS�C	� ����@~�R@l��A�G�C��                                    BxgR��  �          @��H���@���?���A�C޸���@�z�@^�RA�G�C
xR                                    BxgR�~  �          @��R���@Å?s33@��
B�ff���@��R@U�AͅC&f                                    BxgR�$  �          @�p���33@\?�@z�HB�=q��33@�33@;�A�(�C�                                    BxgR��  �          @��R��z�@ʏ\>8Q�?�{B�z���z�@��R@.{A�p�B�                                    BxgSp  �          @����(�@�  >.{?��\B�����(�@���@+�A�p�B�aH                                    BxgS  �          @�p��7
=@��
?\A>=qB��7
=@�{@�=qB��B�                                     BxgS%�  �          @�  �>�R@�\)?�ffA ��B��
�>�R@��
@z�HA���B��                                    BxgS4b  �          @����H@�p�?G�@�ffB������H@��H@G
=A�p�C�                                    BxgSC  �          @��~�R@Ǯ?.{@�Q�B�{�~�R@�p�@J=qA�33B��\                                    BxgSQ�  T          @�{��@��R��ff�Q��C���@�Q�>\@.{C
ff                                    BxgS`T  �          @�����@�Q�+����C�\���@��\?��
A;\)C��                                    BxgSn�  �          @�{��z�@�z�aG���Q�C	�)��z�@�G�?�33Af{C�f                                    BxgS}�  �          @������\@�  >��?�Q�Cp����\@�z�@%A��C��                                    BxgS�F  �          @�  �]p�@�ff?��RA��B��]p�@��@xQ�A��
B�p�                                    BxgS��  �          @�=q�Y��@��?��HA-p�B���Y��@��
@�(�B �B�p�                                    BxgS��  �          @�33��33@θR?E�@�z�B��f��33@�=q@W
=A��B��                                    BxgS�8  �          @�33���@�  ?xQ�@�CJ=���@�=q@VffA�
=C�                                    BxgS��  T          @�(�����@ƸR������B�p�����@�
=@�HA��HC �                                     BxgSՄ  T          @����o\)@���=�?Y��B���o\)@Ǯ@<��A���B��H                                    BxgS�*  �          @��H��Q�@��
�����(�B��R��Q�@�ff@��A�p�C޸                                    BxgS��  �          @����  @����G���C)��  @�(�?�A
�\C�                                    BxgTv  
�          @������@��>�  ?���B�#�����@��H@-p�A���C �3                                    BxgT  �          @�=q��G�@�33?�A;\)B�k���G�@�z�@�z�B��B�                                      BxgT�  �          @�ff�C�
@���E���=qB�q�C�
@Ӆ@
�HA��HB�                                    BxgT-h  �          @����=q@�(���p��O\)B�k���=q@�G�?z�H@�=qB�                                    BxgT<  �          @��H���
@�=q��ff�=�C�����
@�\)?O\)@�C\                                    BxgTJ�  �          @�\���\@���1���p�Cff���\@�p��0�����C�H                                    BxgTYZ  �          @�ff�hQ�@���=p���G�B�G��hQ�@��R?�Q�Ag�
B�=q                                    BxgTh   �          @��R�h��@�ff?B�\@��RB�
=�h��@���@`  A�33B�=                                    BxgTv�  �          @�G����@��������HCL����@vff�;����CL�                                    BxgT�L  �          @�ff��=q@E�i����p�C�q��=q@��\�G��l(�C��                                    BxgT��  �          @��
��(�@>{�;����C�)��(�@{���z��%��C^�                                    BxgT��  �          @��
��
=@]p���\�r=qC���
=@\)�����:�HCB�                                    BxgT�>  T          @��
���H@P�׿.{��Q�CJ=���H@P  ?@  @���Cc�                                    BxgT��  �          @�z����@N{�#�
�#�
C�\���@:�H?���A�RC�{                                    BxgTΊ  �          @�����
@�?�{AZ�HC$�����
?��@#33A�
=C+�
                                    BxgT�0  �          @�33����@+�@3�
A��HCn����?��R@k�A���C)��                                    BxgT��  �          @�Q���G�@A�@\(�A��C^���G�?��@���B�HC'�q                                    BxgT�|  �          @�ff����@��
?���A[�C{����@K�@`  A�\)C(�                                    BxgU	"  �          @����@���?�AqC�����@J�H@g
=A��CL�                                    BxgU�  �          @�����  @mp�@�A�  CO\��  @
=@k�A��C�                                     BxgU&n  �          @�\��ff@��@%A��B�p���ff@�ff@�=qB4��B�
=                                    BxgU5  �          @�=q�E�@Å@B�\A��B�u��E�@�\)@��B6B��                                    BxgUC�  �          @���ٙ�@�=q@HQ�A�  B�  �ٙ�@�33@�(�BC��B܀                                     BxgUR`  �          @�녿��R@�\@Q�A~=qB�\���R@��R@��B"\)B�Q�                                    BxgUa  "          @�{��33@�z�?p��@��B�.��33@ȣ�@�Q�A��
B�u�                                    BxgUo�  �          @�=q���H@�G�@ffA�\)B��)���H@��\@�  B*�\B̀                                     BxgU~R  �          @��@@QG�@�
=BV�BY�@?(�@��HB�W
Ah��                                    BxgU��  �          @�  @N{@Mp�@ǮBW�\B3��@N{>aG�@�Q�B��=@vff                                    BxgU��  �          @��
@hQ�@�  @���B2��BE�H@hQ�?��H@��Bp�A��R                                    BxgU�D  �          A z�@E@��@�{B?33B[�@E?�z�@�Q�B�Aď\                                    BxgU��  �          @��@>�R@�(�@��B0Bjff@>�R@   @��
B}  B��                                    BxgUǐ  �          @�{?�33@�@���B��B��?�33@XQ�@�p�Br\)B�                                    BxgU�6  �          @�=q�(��@��@�AxQ�B�.�(��@�p�@��B�B��
                                    BxgU��  �          @��R�J=q@ڏ\?�=qA z�B�3�J=q@��\@��B�B��                                    BxgU�  �          @��R�~{@�  ?�@��B��)�~{@��@UA�G�B���                                    BxgV(  �          @�Q��u�@�z�?޸RAW�
B���u�@���@��B
��C �q                                    BxgV�  �          @����j=q@�G�@p��A�  B���j=q@Vff@�G�BE��C	�                                    BxgVt  
�          @���3�
@ٙ�@$z�A��B��f�3�
@�  @���B*\)B��                                    BxgV.  �          @�p��'�@Ϯ@eA��B�\�'�@���@��BI�B��                                    BxgV<�  �          @�=q���@�Q�@(Q�A�
=B����@��@�p�B/�RB�W
                                    BxgVKf  �          @�����R@�
=?˅A9��B��
���R@���@�{B  Bؔ{                                    BxgVZ  �          @�ff���@�{>��?�\)B�ff���@�33@`��A�Q�BҞ�                                    BxgVh�  �          @�����
@�=q�u���HB�.���
@�(�@Mp�A�=qB�=q                                    BxgVwX  �          @�Q��z�@��
���H�L  B�W
��z�@��
?ٙ�AJ�RB�W
                                    BxgV��  �          @�(��\)@׮�-p���G�B��
�\)@�  ?�@|��B�k�                                    BxgV��  �          @�(��5�@Ϯ�0����ffB���5�@�G�>\@5B�                                    BxgV�J  �          @�z��G
=@��\�fff��\)B�B��G
=@��H�G��\B�z�                                    BxgV��  �          @�33���H@����33��G�B�ff���H@أ�?��A\)B�k�                                    BxgV��  �          @�����@�z�?�G�AQG�C{����@n{@`��A��\C
�H                                    BxgV�<  �          @������@�������
C������@���?\(�@�33C5�                                    BxgV��  
�          @�ff����@��R�
=��  C}q����@��R?�AY��C�q                                    BxgV�  �          @�=q��ff@�\)�c�
��z�CJ=��ff@�z�?�ffA-C�H                                    BxgV�.  �          @�  ���
@G����\���C h����
@�R=�Q�?J=qC�3                                    BxgW	�  �          @�p���(�@녿�����
C Q���(�@#33    �#�
C:�                                    BxgWz  �          @�  �
=@�G�������B֊=�
=@�z�@9��A���B��                                    BxgW'   �          @�33���@�33�.{���\C+����@�ff?�=qA(Q�C(�                                    BxgW5�  �          @��
��  @�ff�����z�C!H��  @���>��@c33C)                                    BxgWDl  �          @�>�p�@@�
=B�{B��R>�p����R@�p�B��=C�g�                                    BxgWS  �          @�  >�{?˅@ٙ�B�{B��R>�{��{@�G�B��C��)                                    BxgWa�  �          @�Q�@=q?��@��B�ffA�=q@=q��@ʏ\B�B�C���                                    BxgWp^  �          @�@?\)?�z�@�G�BuA���@?\)��=q@��
B{=qC��                                    BxgW  �          @�R@8Q�@^{@���BHz�BH��@8Q�?.{@�(�B�.AS�
                                    BxgW��  �          @��ÿY��@��@���BJG�B�uÿY��?\@�\)B�B�{                                    BxgW�P  T          @���p�@�(�@��HB8(�B鞸�p�?У�@У�B�G�C�3                                    BxgW��  T          @�=q�w�@�@W
=AծB�
=�w�@U@�{B;
=C@                                     BxgW��  �          @���p�@�ff@
=A��C
=��p�@z�H@��HB(�C
s3                                    BxgW�B  �          @�G��mp�@�{?��A33B��H�mp�@��@z�HA�\)B���                                    BxgW��  �          @�\�=q@�p�@B�\A�p�B���=q@�33@��BAB��f                                    BxgW�  �          @��H�=q@��
@�33B1z�B��=q?���@�z�B��C�                                    BxgW�4  �          @��H�333@���@��HB	ffB��333@7�@���Ba�
C@                                     BxgX�  �          @�����@���@2�\A��
B͏\����@�z�@��B>{B׀                                     BxgX�  �          @�33����@�=q@�HA���B�=q����@�ff@��B4�HB̙�                                    BxgX &  �          @�����@�G�@�
A�ffB�8R����@�
=@�  B0Q�B��                                    BxgX.�  �          @�ff��=q@�  ?���Aj=qBř���=q@���@��\B%�BʸR                                    BxgX=r  �          @��׾��@�z�����]��B�L;��@��?޸RAR�\B�G�                                    BxgXL  �          @���\)@��ÿ!G���Q�B�𤿏\)@߮@7�A�=qB�8R                                    BxgXZ�  �          @���>�\)@�G���(��N{B��>�\)@�?�33Ac�B���                                    BxgXid  �          @�=q>�Q�@����z��uG�B��=>�Q�@�z�?���A<��B���                                    BxgXx
  �          @�{�W
=@�  ��R����B�=q�W
=@�\?�{A��B��                                    BxgX��  �          @�ff�U@�ff?�\)AO33B왚�U@���@���B�B��                                    BxgX�V  �          @�  ��\)@��G���  B�(���\)@ٙ�@'�A���B�                                    BxgX��  �          @�\�+�@�����=qB�33�+�@��@;�A���B�
=                                    BxgX��  �          @�(�=L��@�33�������B�33=L��@�  @@  A�G�B�#�                                    BxgX�H  �          @�  �u@��;k����
B��)�u@�p�@L(�A�z�B��                                    BxgX��  �          @�zΐ33@�R�L�;ǮB����33@�@QG�A��HB�G�                                    BxgXޔ  �          @�33���H@߮?�ffA#�
Bπ ���H@��@��
B�
B��                                    BxgX�:  �          @�����@��?�  AQ�B�{��@��@�G�B��B��f                                    BxgX��  �          @�  �u@�(����R�9B�#��u@��
?�G�A<z�B�33                                    BxgY
�  �          @�p��Z=q@��H�e���
B�z��Z=q@��H���H�k�B��                                    BxgY,  T          @��H��\)@�33��  �Y�B���\)@��?�33A�RB�k�                                    BxgY'�  �          @�z����R@�Q�����?�C�����R@�z�?��@��C                                      BxgY6x  �          @���p�@�녿�p��u�C����p�@���?z�@�
=C�                                    BxgYE  �          @�����@�p����_\)B�#�����@��R@A�Cٚ                                    BxgYS�  
H          @�\�xQ�@�G���
=��B�\)�xQ�@��
?�\)Ag�
B�Ǯ                                    BxgYbj  T          @�\�Y��@�\)��33�'�B�{�Y��@��
@4z�A�p�B�\                                    BxgYq  "          @�����@���>L��?\B����@�33@c�
A��
B�8R                                    BxgY�  
�          @�ff�<��@�녾aG���{B�W
�<��@�33@FffA��RB���                                    BxgY�\  
(          @����5@�
=�Ǯ�@  B�ff�5@�33@9��A���B�=q                                    BxgY�  
�          @�33���
@�{�#�
�uB�k����
@�\)@8Q�A���C �\                                    BxgY��  
�          @�����R@�
==L��>ǮC O\���R@�  @5�A���C��                                    BxgY�N  
Z          @�(����H@���?\(�@ϮC�����H@��R@]p�A���C�\                                    BxgY��  T          @�=q���R@��
?���A�HB�  ���R@��@s�
A�C��                                    BxgYך            A=q�n{@�=q�W
=���HB��H�n{@��@O\)A��B�\                                    BxgY�@  
z          A33���H@�{�	���pQ�Cff���H@�\)?�  @�p�C #�                                    BxgY��  �          A�\����@�z��-p����
B�p�����@�p�?��@���B�k�                                    BxgZ�  �          A(���A�R�k��ǮB��3��@�G�@p  A���B��{                                    BxgZ2  �          A�ͿaG�A���  �	B�.�aG�@�33@0��A��B�                                    BxgZ �  T          A���*�HA zῂ�\��33B����*�H@�=q@8Q�A�B��                                    BxgZ/~  
�          A�H�AG�@�G��������B߳3�AG�@أ�@Tz�A��HB��                                    BxgZ>$  T          A(��{@����p��=qB�W
�{@�(�@+�A��B��)                                    BxgZL�  "          A�׿�\)A\)��\)��\Bͳ3��\)@���@8Q�A��\B��                                    BxgZ[p  �          A��p�@�\)����R�HB���p�@�ff?��RA^�RB�
=                                    BxgZj  S          Aff�0��A�
>�@Mp�B����0��@ᙚ@���A�(�B�=q                                    BxgZx�  �          A
{���
Az���H�P  B��R���
@�  @fffA��
B��                                    BxgZ�b  "          A�ÿ��A�H>#�
?��B�uÿ��@�33@��
A�B��=                                    BxgZ�  
�          A	����ffA������BŮ��ff@�p�@:�HA�{Bƨ�                                    BxgZ��  
�          A�H���A녿�p��>�RB�  ���@�
=@z�A���B�L�                                    BxgZ�T  �          Az��\)@��R����,��B��\)@��@33A�\)Bؙ�                                    BxgZ��  �          Ap��@�p��h����G�B���@���@=p�A���B�{                                    BxgZР            A	���'�A �׿����B���'�@��R@(��A��
Bي=                                    BxgZ�F  
�          A�ͿУ�AG�����n�RB��
�У�Aff?�z�AN�HBʣ�                                    BxgZ��  �          Az����A   �  �t��Bͮ����A?�=qAF�HB�\)                                    BxgZ��  �          A�ÿ�33A�Ϳ�G���=qBʊ=��33@���@C�
A�ffB�                                      Bxg[8  �          A��(�A�R��
=�7
=B˸R��(�@�(�@a�AǙ�B�.                                    Bxg[�  �          A{��  A���
=�9��BȨ���  @�@c33AȸRB���                                    Bxg[(�  "          A����ffA�R>���@�B®��ff@��@�{A�=qB�\                                    Bxg[7*  
�          A�Ϳ��@��R@��A���B����@��H@��B2�\B�\)                                    Bxg[E�  g          A�H���@�
=@z�A�B�=q���@�p�@�\)B+Q�B�                                    Bxg[Tv  s          A\)�(Q�@��?�ff@�{BظR�(Q�@�ff@��RBQ�B�W
                                    Bxg[c  	�          A���5�@��>.{?�33B����5�@�@~{A�B�aH                                    Bxg[q�  �          A	G���Q�Ap�?�z�AQ�B�.��Q�@���@��B=qB�=q                                    Bxg[�h  
�          A�R�#�
A (�@�Al��B�.�#�
@���@�ffB-��B�=q                                    Bxg[�  "          A녾�Q�A�R?��\A��B�  ��Q�@��@���B��B�8R                                    Bxg[��  
�          Az�?�A (�@	��Ak
=B�z�?�@��@��B*  B�Q�                                    Bxg[�Z  
�          A33?�Q�A Q�?ٙ�A9��B��
?�Q�@�
=@��HBffB�ff                                    Bxg[�   �          A�R?�@�\)@/\)A��\B��\?�@�ff@��
B;33B�33                                    Bxg[ɦ  
�          A��@�@��H@!G�A��RB�#�@�@�
=@��B4(�B�{                                    Bxg[�L  
�          A��?+�@�
=@UA�{B�Ǯ?+�@�{@���BR�\B��{                                    Bxg[��  �          A	?�RA33@*�HA�{B�k�?�R@�z�@�G�B:�HB�{                                    Bxg[��  �          A	G�>B�\A  @�RAq��B�ff>B�\@���@�B/B��                                    Bxg\>  
�          A�R?�
=@���@6ffA���B�Ǯ?�
=@�{@�  BA33B��\                                    Bxg\�  "          A�\?�@���@#�
A�\)B�aH?�@�@��B:�B�k�                                    Bxg\!�  �          A��+�A  ?���A,(�B�W
�+�@θR@��B=qB��                                    Bxg\00  "          AQ�Q�A(�?ٙ�A4(�B��Q�@�(�@��B (�B��                                    Bxg\>�  "          A(���=qA
�\>#�
?��B�33��=q@�  @��\A�{B�\)                                    Bxg\M|  T          A	����A�?\)@uB�\����@�  @���B��B�aH                                    Bxg\\"  �          A{�
=A�
?�p�A�
B�33�
=@Ӆ@��\B��B�8R                                    Bxg\j�  T          AQ���HA  ?�=q@��B�uÿ��H@�{@�ffB\)Bг3                                    Bxg\yn  
�          A��#�
A
=q������HB�z�#�
@�@~{A�  B��=                                    Bxg\�  T          A�=��
A(�����FffB��=��
@�z�@uA��
B��
                                    Bxg\��  "          A\)?���A
=>k�?���B���?���@�Q�@��A�
=B�\)                                    Bxg\�`  T          A\)�8Q�A�Ϳ�
=���HB�LͿ8Q�A(�@P��A�p�B��f                                    Bxg\�  T          Az���(�@��H�
=q�f=qB����(�@�R?�Q�A3
=B��)                                    Bxg\¬  
�          AG�����@�z��
=�yp�B��R����@�(�?�A��B���                                    Bxg\�R  �          A���z�@����0  ��(�C 33��z�@�G�?J=q@�p�B�B�                                    Bxg\��  
�          A(����H@�����f�HC0����H@���?�ffA��C :�                                    Bxg\�  
�          A
ff����@�z��5���G�CG�����@��>�\)?���C{                                    Bxg\�D  �          A�
��
=@���N�R����C޸��
=@�  �L�Ϳ��Cu�                                    Bxg]�  T          Az���@���j=q��  C	��@�G��+�����C
                                    Bxg]�  �          A����ff@�
=�c�
��ffC�R��ff@�33�L�����\C�H                                    Bxg])6  �          A{��G�@x���@  ��{CJ=��G�@���5��  Ch�                                    Bxg]7�  "          A
=?�G�A�?0��@�  B�
=?�G�@ڏ\@�(�BQ�B�=q                                    Bxg]F�  g          A�H?���A33?fff@�p�B��H?���@ָR@��B�B��                                    Bxg]U(  A          A�H@Q�A?333@�Q�B���@Q�@�\)@��HB�B��                                    Bxg]c�  
(          A�?��A�>��?��B�  ?��@��
@��A�B��)                                    Bxg]rt  
Z          A=q@{A z�h����  B��@{@�{@I��A�z�B���                                    Bxg]�  "          A�
?��A�H��z��4��B�?��@�{@%�A�{B�33                                    Bxg]��  "          A�>�z�A ���������B��H>�z�A�H?�\)AMp�B���                                    Bxg]�f  
�          A=q>���Ap��ٙ��=�B�ff>���@�(�@   A�z�B�=q                                    Bxg]�  g          A=q?�
=@����G��b�\B��?�
=@��
@�Am�B�p�                                    Bxg]��  
�          A	�8Q�@��
�QG����Bި��8Q�A z�?G�@�
=B�p�                                    Bxg]�X  �          A
ff�Y��@��\��33���B�p��Y��@��R���
�?
=B�=                                    Bxg]��  �          A�R��@�����{�!33Bڣ���@�{��  �@  B��f                                    Bxg]�  �          A��6ff@�p����R��B�p��6ff@�  �޸R�C33Bݨ�                                    Bxg]�J  �          A �þ�{@�ff�����#��B�uþ�{@�Q쿮{�5�B��q                                    Bxg^�  T          A�׾W
=@�ff�G���\)B�G��W
=@���?W
=@��B�                                    Bxg^�  
�          A�� ��@��
�Y�����B�Ǯ� ��@��
?\)@y��B�Q�                                    Bxg^"<  
�          A�
�:�H@�R�Q����HB�
=�:�H@�z�?5@�33B܏\                                    Bxg^0�  
(          A33��z�@�p��R�\���\B��
��z�@߮>B�\?�=qB��                                    Bxg^?�  �          A z�����@�Q��G����\B�������@أ�>��?��B���                                    Bxg^N.  �          AG��o\)@��\)���B�Q��o\)@�\)?�{A�B���                                    Bxg^\�  �          A Q����@�G�@uA陚B�u����@:=q@\BJ\)C\)                                    Bxg^kz  �          A��!G�?��@�\B���C���!G����@�RB���C]�                                    Bxg^z   �          Ap��*�H@#�
@��HB}
=CJ=�*�H��@�z�B�ǮCO�q                                    Bxg^��  �          A�\��p�@ ��@�(�B���C� ��p��
=q@��HB�Cc��                                    Bxg^�l  �          AQ��8Q�?:�H@�B��fC%���8Q��S33@�  Bkp�Cd޸                                    Bxg^�  �          A(��7�?�@���B��RC(�3�7��W�@�G�Bg\)Ce��                                    Bxg^��  �          A(��p��>B�\@��HB~\)C1��p���fff@�BP33C_�                                     Bxg^�^  �          A����Q�>��@�ffBv�C0:���Q��^{@��HBL�C\޸                                    Bxg^�  �          A	G���=q?W
=@��Bk��C(�3��=q�9��@�Q�BQ��CU�                                    Bxg^�  T          Aff����
=q@�Bh�C:�)�����\)@���B3�HC^:�                                    Bxg^�P  
�          Ap���(���@���Bl�RC9����(�����@���B7C^޸                                    Bxg^��  T          A������?:�H@�z�Bf(�C+Y������J�H@߮BJ{CUn                                    Bxg_�  �          A33��{?��@��
Bl�C �
��{�%@��HB_�CR:�                                    Bxg_B  �          A���
=?��@�z�BaffC=q��
=��@��B]��CL��                                    Bxg_)�  �          A�R���?W
=@�  BI�C+������1G�@�\)B6{CM                                    Bxg_8�  �          Aff�˅?��@ϮB5�
C.�)�˅�.{@�B"Q�CK&f                                    Bxg_G4  �          A\)��?���@��B�
C%Q������@�(�BQ�C>��                                    Bxg_U�  �          A��\)@e�@���B	33C#���\)?&ff@���B/�C.
=                                    Bxg_d�  �          A	��z=q@�  @Z=qA�
=B�#��z=q@�z�@ʏ\B@{CT{                                    Bxg_s&  �          A���.�RAp����Q�B�8R�.�R@���@��A�z�B�\                                    Bxg_��  �          A�H�E�@���!G���33B�#��E�A�H?�A;�B�=q                                    Bxg_�r  �          A�R�#33A{�����
B����#33A z�@7
=A��B�33                                    Bxg_�  
�          A
=�A	G��(����\)B�u��@�Q�@n{AŅBՅ                                    Bxg_��  �          A�Q�A�þ��
�z�B��f�Q�@�=q@�Q�A�(�B�aH                                    Bxg_�d  �          A=q��G�A
==���?&ffBǽq��G�@�
=@�{A�B��H                                    Bxg_�
  �          A{���A\)?��@�(�Bә����@�G�@��Bz�B��
                                    Bxg_ٰ  �          A33�'�AQ�>�=q?�  B�#��'�@�\)@�  A��HB�                                    Bxg_�V  �          A
=�.�RA
=��=q�޸RB����.�R@�@�  A�Q�B�G�                                    Bxg_��  �          A33�R�\A�H?�\@UB��H�R�\@��@���A��B�{                                    Bxg`�  �          A(��,(�A��?(�@z�HB��f�,(�@�\@�=qB�HB݊=                                    Bxg`H  �          A�H�P  A  >��?�
=B�  �P  @�  @��A��B�Ǯ                                    Bxg`"�  �          A�
����@��H��33��(�B�������@�33@<(�A��B��H                                    Bxg`1�  �          A\)���@��
��ff�"{B�B����@��
@\)A�
=B��f                                    Bxg`@:  �          A
ff�y��@���p��TQ�B�aH�y��@��@G�AYG�B�u�                                    Bxg`N�  �          A{�Z=q@�\�
=�j=qB���Z=q@�z�?�{AMB�                                    Bxg`]�  T          A	���[�@�=q�(��k�B�R�[�@�z�?�ANffB�Q�                                    Bxg`l,  �          A  �\��@���ff�`z�B�=�\��@�p�@�\AY�B�p�                                    Bxg`z�  �          A���Q�@��?�ffAz�B��{�Q�@�  @�z�BG�B�L�                                    Bxg`�x  �          AG��0  A녿ٙ��3�B�u��0  @��
@'�A�p�Bڏ\                                    Bxg`�  �          Ap���\@���R�\��=qBԸR��\A�\?��@�B҅                                    Bxg`��  �          A�
���R@��
�H����33B�k����RA{?�  AQ�B�                                    Bxg`�j  �          A�ÿ���@�ff�Q���  B��Ϳ���AQ�?�
=@�\)B�L�                                    Bxg`�  �          A�� ��A ���'���Q�B֙�� ��Az�?��A;\)BոR                                    Bxg`Ҷ  �          A�\��A��G��S33Bҳ3��A\)@��A�=qB��                                    Bxg`�\  �          A
=�9��Ap�������B�L��9��@�33@FffA�ffB܀                                     Bxg`�  �          A��Q�@���>\@!�B�{�Q�@�(�@��A��B��                                    Bxg`��  �          A��p�@陚�w���33B�.�p�A(�>�33@33B�#�                                    BxgaN  �          A=q�#�
@�{�����  B��H�#�
A	G���z��\)B�B�                                    Bxga�  �          A{�333@ָR���
=B�\)�333Azῦff��
B�aH                                    Bxga*�  �          A\)�(�@����\)�2�B�\�(�A	p���R�y�BЊ=                                    Bxga9@  �          A\)�.�R@�ff�Ǯ�*B�z��.�RA  ����c
=B׊=                                    BxgaG�  �          A��8Q�@������H�.  B�{�8Q�A�\���t(�B��
                                    BxgaV�  
�          A
=�L(�@�33�����QB�u��L(�@����}p���(�B�B�                                    Bxgae2  
�          A����\@�\)��ff�+��B�����\A�R�
=�_�B�\)                                    Bxgas�  �          A=q���
@ᙚ�������BĔ{���
A
=����(�B��                                    Bxga�~  "          A=q�!G�@�����=q��\B����!G�A�ÿ��eB���                                    Bxga�$  
	          A��>�@��������B��3>�Az���Ϳ!G�B��q                                    Bxga��            A�\���
@�p��|����{B�{���
AG�?��@x��B���                                    Bxga�p  �          A(����
A�N�R���HB�ff���
A��?���Az�B�(�                                    Bxga�  T          A�Ϳ��Az��e����\B�Q���A33?�{@���B��3                                    Bxga˼  
�          AQ쿙��@��������
B�Ǯ����A�R�u��33B�L�                                    Bxga�b  �          A�����@�=q��ff��33B�{����A  =�?J=qB�                                      Bxga�  �          Az��{A   �s�
��=qB����{AG�?E�@�z�BȽq                                    Bxga��  
�          A\)�z�@���q��ȣ�BѸR�z�A
ff?333@�\)B��                                    BxgbT  �          Az��@�{������B̔{��A=���?&ffB�ff                                    Bxgb�  
m          A�
���@��
����� Q�BУ׿��AQ쾽p����B�8R                                    Bxgb#�            A��.�R@�ff��
=�33B�B��.�RA�H������\B��)                                    Bxgb2F  T          A�H�C33@׮��(����B�q�C33A�׿J=q��33B�z�                                    Bxgb@�  �          Aff��(�@�����(��\)C aH��(�@�  ��=q�>�RB�                                    BxgbO�  "          A{��\)@�Q���p��(\)C���\)@�p��<(���B��=                                    Bxgb^8  �          A�R��
=@mp����R�(G�Cٚ��
=@�
=�P����CO\                                    Bxgbl�  
�          A=q��(�@�(���p���Cff��(�@�
=����Q�B�B�                                    Bxgb{�  T          A(����@˅���H��B�k����@���\�VffB��f                                    Bxgb�*  
�          A  �j�H@�p��`  ��ffB���j�H@��?   @UB�Ǯ                                    Bxgb��  "          A��	��@��u��ң�B�=q�	��Ap�>��@I��B��)                                    Bxgb�v  
�          A(����
@�=q�z�H���BΊ=���
A(�>��@E�B˳3                                    Bxgb�  T          A{��@���{����B��H��A�׾�=q��p�B�(�                                    Bxgb��  �          A�
��@�(����H��Q�B�uÿ�A�>�@Dz�B�8R                                    Bxgb�h  
�          A\)�˅A��b�\��(�B�G��˅A  ?�ff@�  Bș�                                    Bxgb�  "          A�\����A���0  ��  B�LͿ���A	p�?�\)AB�\B���                                    Bxgb�  T          A�H��A�\�,����G�B�u���Aff?�A;�
BӔ{                                    Bxgb�Z  
�          A�H����A��Z�H����B��쿬��A�?�
=@��HBŅ                                    Bxgc   
�          A{��A Q��8Q���Q�B�z���A�?˅A&{B�.                                    Bxgc�  
�          A{�n{@�
=�AG�����B��n{@��R?���@�Q�B��                                    Bxgc+L  �          A�����@ƸR�L(����\CO\����@�
=>��@&ffC
=                                    Bxgc9�  T          A33����@��
�AG���C�����@��H>Ǯ@�RCu�                                    BxgcH�  �          A\)��z�@�������x��Cn��z�@�?�G�@�
=C�                                    BxgcW>  
(          A�
��33@ʏ\��H�{\)C����33@�?���@�  C0�                                    Bxgce�  �          Az����
@�=q��  ��  B�L����
@���aG����HB��                                    Bxgct�  T          Ap����R@��
�����z�B�����R@��\��(���\)B�G�                                    Bxgc�0  
�          A�����@����ff�W�B�R����@�Q�@	��A\��B���                                    Bxgc��  "          A�H��\)@�z�������B�Q���\)@��Ϳ��
�4  B�W
                                    Bxgc�|  
�          A�����@�����=q�\)C8R����@���(��[�B�                                    Bxgc�"  "          A  ��ff@�(���z��{C)��ff@�(���i�B�8R                                    Bxgc��  �          A����
@�����p�� ��B��q���
@������ϮB��                                    Bxgc�n  �          A33��33@�  ����B����33@��R�
=q�Y��B�u�                                    Bxgc�  �          A\)�e�@��
��  �<�
B��H�e�@��H�Mp���33B�\                                    Bxgc�  
�          A���`��@��\��G��B�B�
=�`��@��X����(�B�=                                    Bxgc�`  �          A����z�@�z�������C�f��z�@˅��33�)�C��                                    Bxgd  	�          A����z�@e�������C����z�@�
=�p��aC�
                                    Bxgd�  �          Ap���z�@�������  CǮ��z�@�
=���_�C��                                    Bxgd$R  
�          A  ����@����G��C
� ����@ڏ\����x(�B�Q�                                    Bxgd2�  �          A
=����@�=q��  �(�C&f����@�p��%��z�B��                                    BxgdA�  "          A�\��p�@�  ��{��C�f��p�@���� ���QG�B�Q�                                    BxgdPD  �          A�����@�p���\)��
B��
����@�(����\�ffB�ff                                    Bxgd^�  
<          A���z�@�������B�\��z�@�녿������B��f                                    Bxgdm�  r          A���p�@��������\C�H��p�@�Q쿸Q��(�B�k�                                    Bxgd|6  �          A
=��
=@�ff������Cn��
=@��Ϳ��H��\C aH                                    Bxgd��  �          A�R��
=@������\��(�B��q��
=@�ff�B�\��p�B�=q                                    Bxgd��  �          A����
@��
�p����ffB��
���
@��\>��?У�B�B�                                    Bxgd�(            A(��~�R@�=q�Y����\)B��~�RA z�?L��@��HB�                                    Bxgd��  r          A�
�e�@�(��E����RB�8R�e�A=q?�p�@��B�p�                                    Bxgd�t  �          A  �w�@�{�n{���B�\�w�AG�>�ff@8��B�(�                                    Bxgd�  �          A���G�@�33�Mp����B�����G�@��R?xQ�@�{B��
                                    Bxgd��  T          A��u@���|(��иRB���uAG�>L��?��
B��
                                    Bxgd�f  �          AG����@�z��5���B�{���@�ff?L��@��
B�z�                                    Bxge   "          A���ff@�Q쿋���RC
8R��ff@���?��RAR�RCs3                                    Bxge�  
Z          A���\@�p�����!�C���\@��H?�A>{C�                                    BxgeX  "          AG����H@�Q���Ap�C�����H@�G�?�p�A5p�C��                                    Bxge+�  T          A�����
@�33�P  ���C ����
@��
>�(�@333B��                                    Bxge:�  
�          A{��z�@�  �x���иRC \��z�@��.{��{B�B�                                    BxgeIJ  �          A=q���H@�\)�'���(�B�8R���H@�(�?�z�@�G�B�\                                    BxgeW�  
�          A�\��@��.�R��33B��\��@�z�?��@׮B��f                                    Bxgef�  h          A=q��=q@�G��^{��G�B�Ǯ��=q@�z�>���@Q�B�.                                    Bxgeu<  
�          Ap���33@θR��p���G�B����33@����R� ��B�
=                                    Bxge��  �          A����@�����33��Q�B�
=���@�  �G�����B�\)                                    Bxge��  �          Aff��z�@ƸR��z���B�\)��z�@��\������B�3                                    Bxge�.  "          Ap����@����tz�����B��)���@�=q<��
>#�
B�(�                                    Bxge��  "          Ap����@θR�n{��Q�B��=���@�R=��
?�\B�{                                    Bxge�z  T          A�\�ʏ\@��^�R��z�C���ʏ\@���@Q�Ax��C	�f                                    Bxge�   "          A��ȣ�@����\)�s\)C\)�ȣ�@߮?�p�@�
=C�                                    Bxge��  "          A�����\@�33��z����C�����\@�녿p�����B�#�                                    Bxge�l  �          A�\���@�z��P  ���CE���@���>�G�@4z�B�8R                                    Bxge�  "          A����{A�*�H��Q�B�3��{Ap�?�A/�
B�ff                                    Bxgf�  
�          A���  @��\�p��X(�C����  @��R?L��@��HC�3                                    Bxgf^  
Z          A=q��33@�33�&ff�{�C����33@���?��@dz�C)                                    Bxgf%  "          AQ��ə�@У��U���
C��ə�@�=q>�(�@!�C��                                    Bxgf3�  "          A=q�n�R@�
=��ff���HB�R�n�RA��(��k�B�L�                                    BxgfBP  T          Az����@��R�����ָRCxR���@�녿0����z�C�H                                    BxgfP�  "          A����=q@�ff�`�����\C)��=q@�ff�\)�Y��C��                                    Bxgf_�  �          A{��z�@����]p����RC)��z�@ᙚ>�?E�C+�                                    BxgfnB  T          A�\�Ǯ@���z�H��
=C0��Ǯ@��;�33�ffC!H                                    Bxgf|�  "          Ap���ff@�=q�s33����C�
��ff@�G��5��\)C
�{                                    Bxgf��  
�          A����z�@�z��:�H��C33��z�@��=��
?   C\)                                    Bxgf�4  "          A�H�\@�ff��  ��z�Cn�\@�  ���G
=C�                                    Bxgf��  T          Ap���p�@�(������  C.��p�@��H������C�                                    