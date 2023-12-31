CDF  �   
      time          *   Conventions       ACDD-1.3, Spase v2.2.3     title         /DSCOVR Magnetometer Level 2 One Minute Averages    id        Doe_m1m_dscovr_s20230118000000_e20230118235959_p20230119020846_pub.nc   naming_authority      gov.noaa.swpc      program       DSCOVR     summary       }Interplanetary magnetic field observations collected from magnetometer on DSCOVR satellite - 1-minute average of Level 1 data      keywords      _NumericalData.ObservedRegion.Heliosphere.NearEarth, NumericalData.MeasurementType.MagneticField    keywords_vocabulary       Spase v2.2.2   
references        �; DSCOVR TIME SERIES DATA AVERAGES ALGORITHM THEORETICAL BASIS DOCUMENT, v2.4; GSE TO GSM COORDINATE TRANSFORMATION ALGORITHM THEORETICAL BASIS DOCUMENT v2.1      metadata_link         �http://www.ngdc.noaa.gov/docucomp/page?xml=NOAA/NESDIS/NGDC/STP/Space_Weather/iso/xml/satellite-systems_dscovr.xml&view=getDataView&header=none    license       Spase.Access Rights.Open   institution       NOAA   source        DSCOVR Magnetometer Level 1    platform      'Deep Space Climate Observatory (DSCOVR)    
instrument        +boom-mounted triaxial fluxgate magnetometer    history       ,DSCOVR real-time telemetry processing system   	algorithm         FDSCOVR MAGNETOMETER LEVEL 1B DATA ALGORITHM THEORETICAL BASIS DOCUMENT     algorithmVersion      B      algorithmDate         
2015-10-15     processing_level      Level 2    processing_level_description      11-minute average using Hodges-Lehmann M-estimator      date_created      2023-01-19T02:08:46.271Z   date_calibration_data_updated         2022-11-28T00:00:00.000Z   time_coverage_duration        P01D   time_coverage_start       2023-01-18T00:00:00.000Z   time_coverage_end         2023-01-18T23:59:59.000Z   time_coverage_resolution      PT1M   creator_name      Doug Biesecker     creator_type      person     creator_institution       DOC/NOAA/NWS/NCEP/SWPC     creator_email         doug.biesecker@noaa.gov    creator_url       http://www.swpc.noaa.gov/      publisher_name         National Geophysical Data Center   publisher_type        institution    publisher_institution         DOC/NOAA/NESDIS/NGDC   publisher_email       william.rowland@noaa.gov   publisher_url          http://www.ngdc.noaa.gov/dscovr/   records_maximum         �   records_present         �   records_data           records_fill         �   records_missing                    time                description       "date and time for each observation     
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
short_name        overall_quality    C_format      %d     units         n/a    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale        7lBx\"�@  �          At  �=p�AL�����R����B�\�=p�Al���3�
�(��Bʞ�                                    Bx\"��  
�          Aq��G
=AAG��\)��\B��f�G
=Ad���b�\�Y�B̊=                                    Bx\"�  T          Apz��I��A<�����33B��)�I��Ab�\�x���pQ�B�{                                    Bx\"�2  *          An�\�=p�A8z��\)��HB�Ǯ�=p�A_�
���R��G�B��f                                    Bx\#�  
�          Am��/\)A:�\����\)B΀ �/\)A`�������{33B��                                    Bx\#~  T          Ap���0��A:ff���33Bή�0��Abff��=q��\)B��                                    Bx\#&$  T          Ap���;�A8(����=qBЊ=�;�A`�����R��{B˅                                    Bx\#4�  �          Aq�.�RA<(��p���HB�.�.�RAc�
��  ���RBɽq                                    Bx\#Cp  
�          Ar=q�2�\A5p��=q�=qBϞ��2�\A`�������B�p�                                    Bx\#R  
�          ApQ��1G�A+�
�"{�)��B��f�1G�AZ�R��33����B��f                                    Bx\#`�  T          Aqp����A0�����%B�u����A^=q��33��(�BǏ\                                    Bx\#ob  �          Ap����HA5��ff�33B����HA`z���{���BǏ\                                    Bx\#~  �          Ap�����A6�\�(����B˞����A`��������=qB�B�                                    Bx\#��  �          ApQ��  A6�\�(��{B�G��  A`����G�����B�(�                                    Bx\#�T  �          Ao33��A3���\�!=qB�p���A_33��  ��ffBøR                                    Bx\#��  	�          An{��A<�����ffBǅ��Ab{�z�H�uB�W
                                    Bx\#��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\#�F              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\#��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\#�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\#�8  	�          Ahz���A1������B�\��AY��
=��\)Bǽq                                   Bx\$�  T          Ag���(�A1p�������B�.��(�AY��G���=qBĀ                                    Bx\$�  �          Ai��\)A0z�����\)B���\)AZ{��Q���{Bƣ�                                   Bx\$*  
�          Ag�
��A1����
��B����AYG���{����B���                                   Bx\$-�  �          AiG����A7��  �\)B�����A]p����\���Bŏ\                                    Bx\$<v  T          Ah���	��A6{���33B�ff�	��A\z���{���BŽq                                    Bx\$K  �          Af�\�333A4�����=qB��H�333AY��x���{33B�\)                                    Bx\$Y�  �          Ae��8Q�A:�\��(��{B�Ǯ�8Q�A[33�Mp��N�HB�Ǯ                                    Bx\$hh  �          Ah�׿���A%��!G��/�
B��H����ATQ���\)��  B���                                    Bx\$w  T          Afff�   A7
=����\B��
�   A[\)�w
=�y�Bę�                                    Bx\$��  �          Afff�:�HAO33������
=B�ff�:�HAa�����B�p�                                    Bx\$�Z  �          Ah���8Q�A3�����RB��8Q�AZ=q������G�B���                                    Bx\$�   T          AlQ��G�A8Q�����=qB�ff��G�A_����
��B�\)                                    Bx\$��  "          Al(���p�A?\)�G��
  B��Ϳ�p�Ab{�c33�_\)B��                                    Bx\$�L  
�          Aj�\�k�A�0(��B�B��k�AN=q��(���Q�B�.                                    Bx\$��  
�          Ak���ffA
�H�<Q��U��B�Ǯ��ffAD����{�{B�                                    Bx\$ݘ  T          An�H�W
=AQ��F=q�`�\B��׿W
=AA��
�\��B���                                    Bx\$�>  
�          Ap  ���
A33�B�H�Y
=B�� ���
AG
=������B�u�                                    Bx\$��  �          ApQ쿊=qA(��Bff�X
=B�
=��=qAG�
�Q����B��)                                    Bx\%	�  T          Anff��(�A��>�H�U
=B�Ǯ��(�AG�� ����
B�33                                    Bx\%0  T          AiG���z�A\)�333�Hz�B�aH��z�AI���\)��B�                                      Bx\%&�  �          Am����p�A�H�5�G\)B��Ϳ�p�AMG��陚��p�B�\)                                    Bx\%5|  �          Ao����AG��733�G��B�p����AP  ���H���HB�.                                    Bx\%D"  
(          Ap  ���
A\)�2{�?��B£׿��
AS�
����ۙ�B�
=                                    Bx\%R�  �          Aq��B�\A!��2{�>�\B��=�B�\AV=q�ۅ�أ�B�u�                                    Bx\%an  "          An�H��33A&{�+\)�7��B��쾳33AW���(���z�B��                                    Bx\%p  �          Anff���A,���!��*�B�G����AZff��(���(�B��R                                    Bx\%~�  �          Ao����A4���G��#�HB��R���A`����\)��p�B���                                    Bx\%�`  T          Aq녿:�HA8���\)��B�B��:�HAc���G���B��f                                    Bx\%�  �          An{���HA6{����   B����HA`(����R���B�                                      Bx\%��  "          Al(���A=�������B�k���Aap��w��s�B¨�                                    Bx\%�R  	�          Am��8Q�AD  ��=q���HB�u��8Q�Ac\)�B�\�=G�B��f                                    Bx\%��  T          An=q�I��AHQ�������z�B�B��I��Ae��#33��B��
                                    Bx\%֞  T          Al���3�
AC����
� \)B��f�3�
Ac\)�Fff�AG�B�ff                                    Bx\%�D  
�          AlQ��Mp�AG
=�����{B���Mp�Ac33�{�{B�z�                                    Bx\%��  "          Ak��Z=qAG�
�޸R��  B҅�Z=qAb{�
=q�\)B��                                    Bx\&�  �          Al���G�A?�
�p���\B�#��G�A`���Y���T��B�                                      Bx\&6  
�          Ak��^�RAD(�����  BӸR�^�RA`���(Q��$��B��
                                    Bx\&�  
�          AlQ��\(�AE���(���  B�=q�\(�Aa��(���$��B�k�                                    Bx\&.�  �          Al  �l��AF�R������B�33�l��Aap��G����B�p�                                    Bx\&=(  "          Ak\)�w�AIG��ҏ\�ԏ\B�(��w�AaG���ff��G�Bҳ3                                    Bx\&K�  "          An{�>�RAA����
=BϮ�>�RAb�R�X���S33B�Ǯ                                    Bx\&Zt  "          Ar�\��G�A;�����G�B��H��G�Ad����z�����B��                                    Bx\&i  �          Ap�ÿ�(�A9�=q��RB��H��(�Ab=q��G�����B��H                                    Bx\&w�  
�          An�H�Q�AA����
G�B�\)�Q�Ad  �qG��j=qB�                                      Bx\&�f  T          Ak��g
=AHQ����H��p�B���g
=Aa����=qBШ�                                    Bx\&�  �          Ah  ����AK����
��Q�B�  ����AW33>���?�B�                                    Bx\&��  "          Aj�\��33AJ�\���R���B����33A]�}p��w�BظR                                    Bx\&�X  
Z          Ah����ffAL����=q��z�B�=q��ffAYp�=�>�B��f                                    Bx\&��  T          Ak
=����ALQ�������B�=q����A_
=�h���c33B�(�                                    Bx\&Ϥ  �          Aj=q�x��AI��=q����B�=q�x��A`Q������Q�B���                                    Bx\&�J  T          Aj�\�]p�AG����H�ޏ\B���]p�Aa��Q����Bϙ�                                    Bx\&��  T          Ak
=�k�AI�������p�BԞ��k�Aap���
=���B�B�                                    Bx\&��  
�          Ajff�n�RAJ�H�ʏ\�̸RB����n�RAaG��˅��  BѮ                                    Bx\'
<  
�          Ai��xQ�AI�����{B�33�xQ�A_\)���H��Q�B�\                                    Bx\'�  �          AiG��w�AJ�\��33���B����w�A_�
��33���B��                                    Bx\''�  "          Aip��j�HAL  ���H��p�B���j�HA`�ÿ�\)���
B�8R                                    Bx\'6.  �          Ai�k�AJ�R��Q����B�W
�k�A`�ͿǮ�ÅB�Q�                                    Bx\'D�  �          Aj=q�l��AJ�\��33��Bԙ��l��Aa���33��\)Bр                                     Bx\'Sz  
�          Aj=q�hQ�AI���\)��(�B�\�hQ�Aa�������B��f                                    Bx\'b   �          Ajff�j=qAHz���z��ׅBԔ{�j=qA`�׿�(���\)B�=q                                    Bx\'p�  
�          Al(��g
=AJ=q��ff���B���g
=Ab�\�   ����BУ�                                    Bx\'l  �          Al(��l��AH���ٙ��ۮB��)�l��Aa��Q����B�aH                                    Bx\'�  T          Aj=q�r�\AF�H��  �ۅB��H�r�\A_��Q����B�L�                                    Bx\'��  �          Ah���^{AE���33���\B�Q��^{A_\)�  �=qB��f                                    Bx\'�^  T          Al(��q�AFff��=q����B��f�q�A`��������B�{                                    Bx\'�  T          Aj�R���AC33�����{B��
���A]� ����BԨ�                                    Bx\'Ȫ  �          Ajff���AG
=�љ���z�B؏\���A^�R���H��{B��H                                    Bx\'�P  
�          Aj=q��  AA�������Bڞ���  A\z��!��33B�=q                                    Bx\'��  �          Ai���=qAC
=���H��(�B����=qA\�����\)B��f                                    Bx\'��  �          Aj=q��  ABff����z�B�p���  A]p��&ff�#�
B�=q                                    Bx\(B  T          Aip����
AAG���=q��Bٞ����
A\  �%�#�B�Q�                                    Bx\(�  
�          Aj{���AB�R��Q����B��H���A]�� ���=qBԽq                                    Bx\( �  T          Ai��w
=A@Q���z�����Bף��w
=A\���;��8��B�G�                                    Bx\(/4  "          Aj=q�o\)A@Q���\)���
B֞��o\)A]G��AG��>�\B�L�                                    Bx\(=�  �          Aj�H�i��AAG���Q����Bը��i��A^=q�A��>�\B�u�                                    Bx\(L�  �          Ak��W�AC
=������  B���W�A`  �AG��=G�B��                                    Bx\([&  
Z          Ai��j=qA?���G���Q�B��j=qA\���G
=�DQ�BѸR                                    Bx\(i�  �          Ai�_\)AA�������B�=q�_\)A]��<���:ffB�G�                                    Bx\(xr  "          Ai��`  AA��G���ffB�G��`  A]���5�4Q�B�ff                                    Bx\(�  �          Aj=q�`  AAG���\)���
B�W
�`  A^{�A��?\)B�W
                                    Bx\(��  T          Ah(��p  A@Q���\����B֮�p  A[
=�,(��+�BҨ�                                    Bx\(�d  �          Af=q��  AC���z���
=B�W
��  AR�\������B�33                                    Bx\(�
  �          Ah  ���\A>�H����G�Bٽq���\AZ=q�4z��3\)B�L�                                    Bx\(��  T          Ag33�}p�A<����33���RB�(��}p�AX���B�\�B=qBԙ�                                    Bx\(�V  
�          Ag33�~�RA<z���33���RB�aH�~�RAX���C33�B�RB�Ǯ                                    Bx\(��  
�          Ah���w
=ABff��G���\)B�8R�w
=A\���(���'
=B�=q                                    Bx\(��  
�          Ag�����AB�R��ff�ܣ�Bؙ�����A[33��
�33B�                                    Bx\(�H  T          Ag
=�x��AB{�أ���  Bׅ�x��A[
=����p�Bӳ3                                    Bx\)
�  T          Ae���x��A>ff��p��癚B�(��x��AXQ��(Q��)p�B�{                                    Bx\)�  T          Ae�0��A3����{BϞ��0��AUG������\B�aH                                    Bx\)(:  T          Af{�   A/�
��H��\B͞��   AT�����H���B�B�                                    Bx\)6�  �          Ae���5�A8(��G��	(�Bϣ��5�AXQ��w��z�\B˨�                                    Bx\)E�  "          Ad(��fffA;���  ���RB�(��fffAW\)�AG��C�
B�                                      Bx\)T,  
�          Adz��g
=A;�
������B�(��g
=AW��B�\�E�B�                                      Bx\)b�  T          AdQ��_\)A;
=��(���G�B�8R�_\)AW\)�J=q�M�B�\                                    Bx\)qx  
�          Ab�H��{A<���������B�#���{AT�������B���                                    Bx\)�  T          Ac���p�A;
=�Ӆ�݅B߅��p�AS33�����RB��                                    Bx\)��  �          Ac���z�A>�H���\��p�B����z�AR�R��33��z�Bި�                                    Bx\)�j  �          Ac���G�A@�������  B�W
��G�AQ��xQ��{�B��f                                    Bx\)�  �          Ac�����AAG����\���B�33����AQG��fff�h��B��)                                    Bx\)��  "          AdQ���  A?\)��  ���B�aH��  AQp�������B�\                                    Bx\)�\  "          AdQ���{A@����33����B�\��{AS�����ffB��)                                    Bx\)�  �          Ac���Q�ABff��ff��p�B�Ǯ��Q�AR�H��G����B�z�                                    Bx\)�  �          Ad  ����A@��������B�=����AR�R��=q���\B��H                                    Bx\)�N  T          Ab�\��(�A?�������B�aH��(�AP�Ϳ���  B��
                                    Bx\*�  "          Aa����A<Q������ffB�����AL  �u�z=qB�\)                                    Bx\*�  T          A`Q���{A=p�����z�B�W
��{AL�׿aG��eB��                                    Bx\*!@  T          A_\)����A=���33���\B�������AL�ͿL���S33B�q                                    Bx\*/�  
�          A`����
=A2�H��33��\B���
=AM�J=q�O�
B�L�                                    Bx\*>�  �          Aa����(�A,Q��陚��B�
=��(�AH���_\)�ep�B�p�                                    Bx\*M2  
�          AaG����A0z���{��\)B�ff���AM��c33�i�Bۏ\                                    Bx\*[�  "          Aap�����A333����B�(�����AM���Fff�K�B�                                    Bx\*j~  �          A^�H���\A1����ff��G�B�L����\AI��&ff�,��B��                                    Bx\*y$  �          A]���(�A7���\)�ՙ�B�����(�AM���33Bۣ�                                    Bx\*��  
�          A]p�����A6ff����ͅB�{����AK\)��
�HB�#�                                    Bx\*�p  �          A\(��
=A�H�s33��z�C33�
=A)녾�׿�p�CJ=                                    Bx\*�  h          A[����
A'
=���R��  B����
A;33�Q��(�B�=q                                    Bx\*��  �          Aa?�
=A"�R�=q�,ffB�=q?�
=AJ�H���R��B��                                    Bx\*�b  T          AaG�@�A���R�-�B�8R@�AH  �����ˮB�L�                                    Bx\*�  
�          Ab{@1G�A�����.��B�u�@1G�AE��p��ϙ�B��q                                    Bx\*߮  �          A`��@0  A�R�p��+��B�  @0  AF�R��Q���ffB�                                    Bx\*�T  �          Ac33@�RA#33���*��B�\)@�RAK
=��ff��ffB�{                                    Bx\*��  �          Af=q?�
=A'33�33�)��B��?�
=AO33��
=��  B�{                                    Bx\+�  T          Ad��?��A+
=��\�$��B�G�?��AQp����
���B�Ǯ                                    Bx\+F  �          Ad��?�\)A((��=q�)�B��)?�\)AO���������B�(�                                    Bx\+(�  �          Aep�?�G�A+��ff�#�B��?�G�AQ�����
���\B���                                    Bx\+7�  �          Ad(�?�A-���!G�B�u�?�AR�H������\B�p�                                    Bx\+F8  "          Adz�>.{A-��Q��"ffB�.>.{ARff��\)����B��                                     Bx\+T�  �          AhQ�@333A((���%B��@333AO\)�������
B��{                                    Bx\+c�  T          Ah��@�A*{��%��B�8R@�AQ���(����RB��                                    Bx\+r*  �          Ah��@�A*ff��\�&Q�B�Ǯ@�AQ������(�B�8R                                    Bx\+��  ]          Ah��@�A+�
�p��$��B�{@�AR�\���\��33B�
=                                    Bx\+�v  
�          Ahz�@�A+33����%ffB���@�AQ����
��z�B���                                    Bx\+�  
�          Ag�?У�A+
=��&�B��?У�AQ��(���{B��
                                    Bx\+��  �          Ae��?��A+\)��
�%�B��?��AQp��������B���                                    Bx\+�h  T          Ae�?��A+����%{B��H?��AQ����Q����B�\                                    Bx\+�  �          Aep�?���A/\)�=q�z�B�  ?���AS���(���ffB�aH                                    Bx\+ش  
�          Ad��?�\)A+��  �!\)B�.?�\)APz������G�B���                                    Bx\+�Z  T          Ab�\?���A*�\�� �
B��H?���AN�\���R��{B��
                                    Bx\+�   
�          A`(�?�ffA&�R���$��B�\?�ffAK��������B��                                    Bx\,�  �          A`��?޸RA,���G���B��3?޸RAO\)��p����\B��3                                    Bx\,L  �          Aa@	��A.ff�
�H��B���@	��AP(���  ��{B�33                                    Bx\,!�  "          A`��?�p�A0���\)�  B���?�p�AQG���  ��(�B�                                    Bx\,0�  �          A]?@  A4����\)���B�aH?@  AR�\��\)��ffB�u�                                    Bx\,?>  
�          A]�?�Q�A3������RB�ff?�Q�AQG���{��G�B�p�                                    Bx\,M�  
Z          A]G�?��RA4z�����(�B���?��RAQ�������G�B��                                    Bx\,\�  �          A^�H@��A2�H� z���
B�L�@��AP�����H��G�B�\)                                    Bx\,k0  
�          A^�H?�  A3�
� Q��B�G�?�  AQ���=q����B�Ǯ                                    Bx\,y�  �          AZ=q?У�A'��	��p�B�Q�?У�AH����(���Q�B�8R                                    Bx\,�|  T          AW�
?�p�A*{� ���B��?�p�AH���������
B�\                                    Bx\,�"  T          AU��?�\)A'\)��R�\)B�Ǯ?�\)AF�\��
=��  B��3                                    Bx\,��  
�          AV=q?�{A(  �G����B��H?�{AF�R��(����B��                                    Bx\,�n  ]          AU��?��HA"=q�	G�� (�B�k�?��HAC\)��
=��Q�B��R                                    Bx\,�            AT��?�z�A#
=�\)�(�B�  ?�z�AC���33���\B�#�                                    Bx\,Ѻ  �          AP��=�Q�A%�����{B���=�Q�AC33���
��  B�#�                                    Bx\,�`  �          ALQ�ٙ�A+��ָR����B�  �ٙ�AC��Q��nffB��
                                    Bx\,�            AF�R�޸RA0����=q��33B�녿޸RAB{��{�
=qB�\)                                    Bx\,��  �          AC�
��A/\)�����\BɊ=��A>�R�\��B��
                                    Bx\-R  T          AEp���A5�����=qB�.��AA녿s33��B���                                    Bx\-�  T          AC
=��{A.�R���\�ͮB�Q��{A@  ��Q����Bř�                                    Bx\-)�  �          APQ����A0z���  ��B̙����AG
=�B�\�W�
B��H                                    Bx\-8D  "          AQ��-p�A1G����
��\)Bπ �-p�AG33�9���Mp�B̔{                                    Bx\-F�  "          AMG��#�
A2{��(���{B���#�
AE�����.�RB�p�                                    Bx\-U�  
�          AL(��A�A1���=q��G�BҞ��A�AC33�
=q�(�B��f                                    Bx\-d6  �          AO
=�VffA3����H��{B�8R�VffAE�����
B�Q�                                    Bx\-r�  �          AQ���dz�A2=q�����=qB׊=�dz�AE���R�.=qB�8R                                    Bx\-��  �          AN{�J=qA/�����G�B���J=qAC��#33�6�RB�                                    Bx\-�(  
�          APQ��vffA3�
�������B��)�vffADQ��{��B��H                                    Bx\-��  �          AO�����A333��p����
B۔{����AC33���
��z�B؊=                                    Bx\-�t  
�          AFff����A*ff��  ��{B������A7�
��ff����B�\                                    Bx\-�  
�          A@Q��L��A(Q����\����B����L��A7\)��z���{B�.                                    Bx\-��  
�          ALz��!G�A-��{��Q�B�#��!G�AB�\�8Q��P��B�k�                                    Bx\-�f  T          AL���aG�A3�
�����{B��aG�AC\)�ٙ����
B�#�                                    Bx\-�  �          AE�����A3����2�RBݳ3����A7
=?^�R@��\B���                                    Bx\-��  �          AJ�H���A5���h����p�B�\���A>�\���
���HB�8R                                    Bx\.X  �          AMp��UA8(���=q��
=B�aH�UAEG���
=����B�W
                                    Bx\.�  �          AN�\�%A3�
��G���G�B�  �%AFff���-�B˞�                                    Bx\."�  �          ALQ�����A6{�g���G�B�\����A?
=���
����B�Q�                                    Bx\.1J  �          AN�\��  A;33�{��\B�=q��  A=?�{@��B�3                                    Bx\.?�  T          APQ���Q�A<���0���D  B����Q�AA?(�@*�HB��                                    Bx\.N�  �          AP����p�A>{�0  �B=qB����p�AB�H?#�
@2�\B���                                    Bx\.]<  �          AO\)��33A<  �:�H�Pz�B�����33AA��>�
=?��Bݨ�                                    Bx\.k�  �          AH����A5��-p��G\)B��f��A:{?   @33B��
                                    Bx\.z�  �          AH����33A6{�P���qG�Bۣ���33A=G����
����B�8R                                    Bx\.�.  J          AJ=q�vffA333��G���G�B���vffA?
=�������B׸R                                    Bx\.��  T          AK33��
=A.�H��p���RB����
=AC33�=p��X  B��                                    Bx\.�z  T          AI�9��A3
=��\)��z�B��9��AA녿�  � (�B��H                                    Bx\.�   
�          AJ�R�1G�A4Q���G����
Bϙ��1G�AC\)��ff��HB͊=                                    Bx\.��  �          AM�O\)A5��33����B����O\)AD  ������ffBљ�                                    Bx\.�l  �          ANff�|��A:�H�r�\����B�W
�|��ADQ�   �{Bר�                                    Bx\.�  �          ANff��G�A;
=�j=q��B�{��G�AC�
�\��
=B�z�                                    Bx\.�  T          AO33��Q�A<  �k���  BٸR��Q�AD�þǮ�޸RB��                                    Bx\.�^  �          AN�R�@��A=����(����BЅ�@��AHQ�O\)�e�B�
=                                    Bx\/  T          AM��A�A<���������\B��H�A�AF�H�=p��S�
B�p�                                    Bx\/�  �          AMp��J=qA;33��\)��  B�=q�J=qAF�\�xQ����\BЙ�                                    Bx\/*P  �          AM���RA;�
������HBɊ=��RAIG�������\)B�#�                                    Bx\/8�  T          AIp���A0��������p�B�#׿�AC
=�*�H�C�
BÏ\                                    Bx\/G�  �          AI��\��A5���G����RB���\��A?��h�����B�(�                                    Bx\/VB  �          AJ�H��33A6�\�
=�G�B����33A9�?xQ�@�(�B�\                                    Bx\/d�  �          AJ=q��
=A5녿�ff�33B�Q���
=A7
=?��R@���B�
=                                    Bx\/s�  �          AK�
���
A8  �   ���B������
A9�?�=q@�33B�                                    Bx\/�4  �          AK�
���RA8�����p�B�u����RA;33?xQ�@�33B��                                    Bx\/��  �          AI��A5�����!G�B����A8��?Tz�@r�\B�B�                                    Bx\/��  �          AG�
���RA333�7��Tz�B�{���RA8��>\)?#�
B�L�                                    Bx\/�&  �          AG���
=A6�H��\�(��B޽q��
=A:{?@  @\��B�\                                    Bx\/��  T          AH(���\)A7\)�  �%B޳3��\)A:�\?G�@fffB�\                                    Bx\/�r  �          AH  ���\A6�H�����B߳3���\A9��?^�R@�  B�#�                                    Bx\/�  �          AH������A733���2�RB�(�����A;
=?
=@,��B�W
                                    Bx\/�  �          AJ{���A7\)�>{�Z{B�=q���A=p�=u>�z�B�                                    Bx\/�d  T          AI�����A6=q�AG��]G�B�aH����A<z�#�
�#�
B�\                                    Bx\0
  �          ALQ���
=A7��S�
�pz�Bފ=��
=A?
=��\)��  B�                                    Bx\0�  �          AP  �n�RA;���{��ffB�L��n�RAF�\��=q��  B�z�                                    Bx\0#V  �          AP���c33A9�������RB�
=�c33AG33���H����B��)                                    Bx\01�  �          AQ��8��A:=q�������B���8��AH���33�z�B��                                    Bx\0@�  �          AO�
�-p�A5����ff��Q�B��H�-p�AF�R�*=q�=G�Ḅ�                                    Bx\0OH  �          ARff���A:=q���H��\)B����AJ�R�\)�.�HB��)                                    Bx\0]�  "          AR�\�ffA:{�����Џ\B�k��ffAK\)�,(��<��BƸR                                    Bx\0l�  �          AQp���=qA=�������HBř���=qALQ���
�G�B�W
                                    Bx\0{:  �          AS33���\A<Q���33����B��)���\AM�/\)�@  B�
=                                    Bx\0��  �          AS33�\)A;�
�������B�uÿ\)AM�:�H�L��B���                                    Bx\0��  �          AT  �
=qA:{������\B�LͿ
=qAM��Mp��`z�B���                                    Bx\0�,  �          AVff��=qA<����G���Q�B��{��=qAP  �L(��\z�B�W
                                    Bx\0��  �          AU=L��A>{�Å���
B��=L��AP(��@���P(�B��{                                    Bx\0�x  T          AU�>��RA>{�Å�ٮB�>��RAPQ��AG��P��B�G�                                    Bx\0�  �          AV{?L��A>{��33���HB�W
?L��AP(��@���P(�B�                                      Bx\0��  �          AV�H��\)ADz���p�����B��׽�\)AS��G���
B���                                    Bx\0�j  �          AU��?^�RAC�
��
=��Q�B��H?^�RAR=q��z�B�p�                                    Bx\0�  �          AT��>���AC\)������
B�.>���AQ��Q���
B�\)                                    Bx\1�  �          AU��>�G�AC33���
��  B��H>�G�AR{�G����B�(�                                    Bx\1\  �          AU�����AC���
=��z�B�녿��AQ�Q��33B�B�                                    Bx\1+  �          AT(���Q�A@Q���  ��z�B�W
��Q�AN�H�p��B�\                                    Bx\19�  �          AR=q�"�\A7
=�������
B�
=�"�\AHQ��@  �S\)B��                                    Bx\1HN  �          AQ��C�
A7�
���\��
=B��
�C�
AF�H�(��,z�BϮ                                    Bx\1V�  T          AN�H�E�A?�
�hQ����RB����E�AH(��.{�AG�BϨ�                                    Bx\1e�  �          AM���c�
A=G��e����BՀ �c�
AEp��0���Dz�B�.                                    Bx\1t@  �          AJ=q�q�A5��������\B�Ǯ�q�A?�
��p����\B���                                    Bx\1��  �          AF{�w
=A.�\��������B����w
=A:=q���H���RBس3                                    Bx\1��  �          AP�����A>�H���\��
=BʸR���AJ�R��z���\B�k�                                    Bx\1�2  �          AQG����
A>�\������  B����
AL(��Q��{B��f                                    Bx\1��  �          APQ���A>�\��  ����B�{��AJ{��{���
B���                                    Bx\1�~  �          AO
=�Z�HA;33��G���33BԔ{�Z�HAE녿��H�ϮB��H                                    Bx\1�$  T          AP���J�HA?
=���R��ffBѽq�J�HAIp���=q���B�G�                                    Bx\1��  
�          AQ��B�\A?
=�����(�BЊ=�B�\AJff��\)��\B�                                      Bx\1�p  �          APQ��O\)A?33�}p����B�\)�O\)AH�׿�{����B���                                    Bx\1�  �          AJ�\�/\)A8Q���\)���
BθR�/\)AC�
��(����\B�33                                    Bx\2�  �          AK
=�(Q�A7�
��{��\)B͸R�(Q�AD(���Q����B�.                                    Bx\2b  �          AL���#33A8���������B��H�#33AE���
=��B�Q�                                    Bx\2$  �          AK�
�(Q�A8����\)��Q�Bͳ3�(Q�AD�ÿ��R�  B�#�                                    Bx\22�  �          AH���*=qA4�������B�p��*=qAAG��z���
B���                                    Bx\2AT  T          AL(��333A8  ��G���ffB�W
�333ADz����=qBͣ�                                    Bx\2O�  "          ANff�EA8Q���(����
B����EAE��
�H��HB��                                    Bx\2^�  �          APz��tz�A4��������
B�Q��tz�AC
=�%�7
=B�Ǯ                                    Bx\2mF  �          AP������A3\)��33��G�Bۅ����AA��-p��?�
B�                                    Bx\2{�  �          AQ�����A2�\��\)��=qB۸R����AA���7
=�J�\B��)                                    Bx\2��  T          AQp��o\)A5�����p�B�ff�o\)AD(��*=q�;�
B��f                                    Bx\2�8  �          AQ��h��A7
=��z����B�Q��h��AD���   �0(�B�                                      Bx\2��  T          AP���k�A8�������
=B�L��k�AEp��	���  B�8R                                    Bx\2��  �          AP���w�A9��  ��p�B����w�AD�ÿ����=qB��)                                    Bx\2�*  �          AP���Z�HA:�\�������BԸR�Z�HAF�H�	���\)B���                                    Bx\2��  �          AP���C�
A<z���p����B���C�
AHQ��   ��B�z�                                    Bx\2�v  
�          APQ��i��A733��Q�����B�aH�i��AD(��=q�*�RB�(�                                    Bx\2�  
�          APz��R�\A9������Q�BӸR�R�\AE��Q��(z�B�                                    Bx\2��  �          AO�
�c33A8Q����\����B�B��c33AD���\)��HB�=q                                    Bx\3h  �          AN=q��=qA5��Q���{B�k���=qA@�ÿ�p���B�G�                                    Bx\3  T          AN{����A2�H���
��ffB�������A>�\�Q����B܏\                                    Bx\3+�  T          AM������A2�H��p�����B�\����A=녿�Q��33B�                                    Bx\3:Z  T          AN=q����A7
=������\B�  ����AA������   B�
=                                    Bx\3I   �          AK���(�A2{�����
=B�.��(�A;�
��\)��\B�\                                    Bx\3W�  "          ALQ��Dz�A8Q�������ffB��H�Dz�AC�� �����B�B�                                    Bx\3fL  
�          AK��9��A8z���������B�B��9��AC�� ���G�BθR                                    Bx\3t�  �          AJ�H�G�A8������=qB�=q�G�AB�\��
=��(�B�                                    Bx\3��  "          AJ=q�5�A9��������Bϊ=�5�AC33�޸R���B�(�                                    Bx\3�>  
�          AG��7
=A7
=��33��p�B�
=�7
=A@�׿����=qBγ3                                    Bx\3��  T          AE�7�A4�����
��BЏ\�7�A>ff�ٙ���p�B�(�                                    Bx\3��  T          AEp��:�HA4���\)��
=B���:�HA>=q�˅��(�BϏ\                                    Bx\3�0  �          AD���W�A4z��a�����B�=q�W�A<Q쿓33���
B���                                    Bx\3��  �          AF�\�HQ�A5���z�H��33B��)�HQ�A>�\�\��G�B�u�                                    Bx\3�|  
�          AEG��;�A5G��vff���B�  �;�A>{��(���33BϸR                                    Bx\3�"  
�          AB�R�J=qA2�\�k�����Bә��J=qA:�H������33B�G�                                    Bx\3��  T          ABff�N�RA2�H�^�R��\)B�8R�N�RA:�\��z���\)B���                                    Bx\4n  �          A>=q�C�
A/��W����B�(��C�
A6�R������=qB���                                    Bx\4  @          A)���{�A���{�B�RB��{�A�׾�=q��p�B߸R                                    Bx\4$�  
�          Ap����\A
{�����{B�=q���\Az�>#�
?h��B�z�                                    Bx\43`  
�          A{���
A�Ϳ�����B�p����
A33=�Q�?�B��                                    Bx\4B  �          A�R�vffA33��{�B����vffA	��=�Q�?�B�8R                                    Bx\4P�  {          A(��~{A  ��=q�Q�B�aH�~{A>��
?�p�B��
                                    Bx\4_R  T          A�x��A�R��G����RB垸�x��A(�>�
=@#�
B�.                                    Bx\4m�  �          A�\�h��A�ÿ��
� ��B�W
�h��Aff>�Q�@G�B��H                                    Bx\4|�  T          Aff�p��A (��s33��z�B�G��p��A ��?�R@�Q�B��                                    Bx\4�D  T          A���b�\A (��Y������B�3�b�\A Q�?8Q�@�B㞸                                    Bx\4��  �          A�aG�AG��aG���B���aG�A��?333@�Q�B�
=                                    Bx\4��  
�          A{�\��A�k����B�#��\��A=q?&ff@��RB�                                    Bx\4�6  
�          Ap��u@�ff��\)���
B癚�u@�33?�  A=qB��                                    Bx\4��  T          A���\(�A�ͿQ���{B�{�\(�A��?G�@��RB�\                                    Bx\4Ԃ  �          A�\��A�H>�33@�B�R��A\)?�Q�A>ffB��                                    Bx\4�(  �          A�R�g�Aff���P  B➸�g�A��?�=q@�  B��H                                    Bx\4��  
�          A�\�UA
=�����\)B�ff�UA�
?�\@S�
B�(�                                    Bx\5 t  T          A��@  AQ쿋����B���@  AG�>�@EBۮ                                    Bx\5  T          A��4z�A(��˅�%p�Bٮ�4z�A�\���
���B�
=                                    Bx\5�  T          A��\)Az���R�O33BՅ�\)A  ��
=�,(�BԮ                                    Bx\5,f  �          Az��*�HA�z��YB�k��*�HA�����^{B�p�                                    Bx\5;  T          A\)�;�A Q��z��0z�B�.�;�A
=�#�
��ffB�k�                                    Bx\5I�  
Z          A��Z�H@�33��(��5B���Z�HA zᾀ  ��z�B�(�                                    Bx\5XX  
�          A\)�1�AG���p����B���1�A�<��
>#�
B�W
                                    Bx\5f�  �          A���Q�A
=�������
B�aH�Q�AQ�>�33@�B��                                    Bx\5u�  �          A����HA녿�33���B�L���HA
=>\@   B�
=                                    Bx\5�J  �          A(����Aff��\)��(�B�L����A�>��@*=qB�{                                    Bx\5��  �          A��˅A����
�\)B�G��˅A	�>��?ٙ�B�\                                    Bx\5��  
�          A	����A�\��
=�4��B�p����AG��L�Ϳ�=qB��H                                    Bx\5�<  "          A��  @�{��{�K33BӞ��  Aff��
=�5B��)                                    Bx\5��  "          A	��!G�A z��  �!G�B����!G�A�R�L�;�{B�=q                                    Bx\5͈  
�          A��
�HA�H�ff�^�RBѽq�
�HA�H��R��33B��                                    Bx\5�.  T          AG���{A�\�.�R����Bͽq��{A(���G���B̳3                                    Bx\5��  
�          A�
��=qAff�\����Q�B����=qA
{��(��JffB�Ǯ                                    Bx\5�z  �          A\)���HA�
�Q���  BĨ����HA
�H��ff�9p�B���                                    Bx\6   �          Aff�
�HAp��k���  B���
�HA	����R�aB�\)                                    Bx\6�  T          A=q��=qA (���  �ϙ�B��Ϳ�=qA	G��#�
��{B�#�                                    Bx\6%l  T          A�ÿ�ff@�����z���B�#׿�ffA  �.�R��{Bȣ�                                    Bx\64  �          A
=����@�(��e����RB�\����A{���d  B̔{                                    Bx\6B�  T          A�\��\@�R��
=��(�B���\A�H�����
B̮                                    Bx\6Q^  �          A�R�ٙ�@������B�W
�ٙ�A�H�Dz�����B�k�                                    Bx\6`  �          A�����@�������(�B�����@���r�\�Ώ\Bˏ\                                    Bx\6n�  T          A���p�@�p������B�\��p�@���s33��ffB�k�                                    Bx\6}P  
�          AQ��Q�@������ ffB�\)��Q�@��H�Y����z�B��
                                    Bx\6��  �          A
{�=q@�
=�c�
��p�B��=q@��R���t��Bճ3                                    Bx\6��  �          A(��N�R@�33�(Q�����B�p��N�R@������{Bᙚ                                    Bx\6�B  �          A=q���\@ָR�#33���B�����\@�G�����B�33                                    Bx\6��  �          A�R��(�@�  �{��ffB����(�@�=q��=q�p�B�                                    Bx\6Ǝ  T          A33���\@�p��p��l��B�=q���\@����\�׮B�ff                                    Bx\6�4  �          A	��S�
@�
=�!����B����S�
@��ÿ���
�RB�{                                    Bx\6��  �          AQ��l��@�33�A���B�p��l��@�  ���>{B�{                                    Bx\6�  �          A���w�@����?\)���B�  �w�@�p����
�;�B�{                                    Bx\7&  �          A
=����@�(��HQ���
=B�R����@�G����R�T  B�Ǯ                                    Bx\7�  �          A\)���@�
=�E����RB��H���@�(���(��W�B��                                    Bx\7r  
�          A	G���z�@�(��O\)����B�\��z�@���	���hz�B�Ǯ                                    Bx\7-  �          A	G���33@�\�!���33B�#���33@�z῰���Q�B�
=                                    Bx\7;�  
�          A  �z=q@�
=�6ff���B�z=q@�\��p��;�B�.                                    Bx\7Jd  �          A���5@�=q�.�R��p�B�#��5@���\�$(�Bۊ=                                    Bx\7Y
  
�          A	��N�R@�
=�'
=��Q�B��N�R@�G�����B�\                                    Bx\7g�  
�          A
�R�HQ�@�ff������B�8R�HQ�@����p��
=B�Ǯ                                    Bx\7vV  �          A33�Y��@��H�$z���p�B�=q�Y��@�z`\)�G�B�{                                    Bx\7��  
�          A��G�@�\)�$z���33B��G�A z΅{��Bހ                                     Bx\7��  
Z          A
�H�1G�@���33�v�\B��H�1G�A녿�����z�Bٽq                                    Bx\7�H  �          A(��dz�@�p��ff�e�B�k��dz�@���s33��ffB�
=                                    Bx\7��  T          A=q�l(�@��������BꞸ�l(�@�p����\�\)B��)                                    Bx\7��  T          A��hQ�@�p��&ff��p�B��hQ�@�\)��  �#\)B�                                    Bx\7�:  
�          A�R�p��@���%�����B��p��@��
��  �$(�B�
=                                    Bx\7��  �          A=q����@�\)�����Q�B�ff����@�Q쿮{�B�B�                                    Bx\7�  "          A33���@�p��#�
��G�B������@�\)��ff�)�B�z�                                    Bx\7�,  T          Az���{@ָR������B��)��{@߮��{�ffB��R                                    Bx\8�  �          A�����@�{�.{���B�33����@أ׿�  �>�HB��                                     Bx\8x  �          A���@�z����(�B�\���@���������B���                                    Bx\8&  T          A����G�@��
�
=��  B��)��G�@�zῴz����B��\                                    Bx\84�  "          A�\���R@ʏ\�(����C ����R@��
��G��%G�B��3                                    Bx\8Cj  T          Aff��\)@����z����RC�H��\)@�p�������
C��                                    Bx\8R              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\8`�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\8o\              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\8~              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\8��  �          A����33@У��G��~�\B�W
��33@��ÿ����B�L�                                   Bx\8�N  "          A����G�@�Q��=q��G�B�����G�@�G����R�%p�B���                                   Bx\8��  �          A������@�Q��G��~{C=q����@ȣ׿���C\                                   Bx\8��  T          A����Q�@�(������YC#���Q�@��H�������C!H                                   Bx\8�@  
�          A����
=@�z����S
=Cff��
=@��H���
��C}q                                   Bx\8��  "          A������@�Q���H�=G�C������@�{�aG�����CǮ                                    Bx\8�  �          A����\@�(���=q�J�HC���\@�=q�z�H��G�C(�                                    Bx\8�2  T          A����{@���z��U�C��{@�(������(�C!H                                    Bx\9�  S          A(����@ȣ��G��b{C �����@�\)��z��p�B�=q                                    Bx\9~  T          A  ��@�G����d��C 
��@�  ��
=�  B�W
                                    Bx\9$  
�          A���(�@������k
=B��q��(�@�  ���R�
�\B��
                                    Bx\9-�  T          A  ���@�����d(�B����@��
�����{B�k�                                    Bx\9<p  �          AQ���ff@�(���j�RB����ff@�33���H��\B�                                      Bx\9K  T          A�����@�z��ff�I�B�=q���@�녿c�
��ffB���                                    Bx\9Y�  
�          A  ��  @��
��z��9G�B� ��  @��ÿB�\��  B�W
                                    Bx\9hb  
Z          A\)���@�z���H�@��C����@�녿h�����
CL�                                    Bx\9w  T          A�����@��������C)���@Å���z�HC��                                    Bx\9��  
�          A�����\@�p���33�7�CǮ���\@��\�k���33C��                                    Bx\9�T  �          AQ����
@��ÿ����K33C8R���
@��R������
=CT{                                    Bx\9��  "          A\)��p�@�{�У��7
=C�f��p�@�33�fff��G�C)                                    Bx\9��  "          A
=��\)@�33��z��;33C����\)@��׿s33����C��                                    Bx\9�F  "          A(���Q�@����
=�<Q�C	�q��Q�@��\��  �޸RC	�                                    Bx\9��  T          A33��=q@�(������\)C
h���=q@�\)���P  C	�f                                    Bx\9ݒ  �          A33��@�
=��33�p�C	Y���@�33�8Q���G�C��                                    Bx\9�8  �          A  ���R@�Q쿳33�Q�C	@ ���R@�z�8Q���\)C�
                                    Bx\9��  �          AQ����@�G��k����C�f���@Å�8Q쿣�
CW
                                    Bx\:	�  
Z          A
=��G�@��R��(��C
=C0���G�@�(���=q���
C
G�                                    Bx\:*  �          @��R��  @vff�0����
=CT{��  @������\)C\)                                    Bx\:&�  �          @��
���
@�\�n�R����C�����
@.{�[���G�C#�                                    Bx\:5v  �          @�R��p�@9���N{����C�f��p�@P���7
=����C{                                    Bx\:D  �          @���=q@"�\�0����  Cff��=q@6ff�(���33C��                                    Bx\:R�  �          @�\)��z�@E�����Cc���z�@S33��z��\Q�C�                                    Bx\:ah  �          @��H����@C33� �����\C.����@P�׿���]�C�\                                    Bx\:p  �          @�33��=q@\)�&ff���RC�{��=q@1G��33����C��                                    Bx\:~�  
(          @ۅ��z�@#33��H����C����z�@3�
�
=���C�                                    Bx\:�Z  �          @�ff��(�@,(��   ���HCs3��(�@=p�����G�CE                                    Bx\:�   
�          @�z���\)?ٙ�����p�C$+���\)?�Q�����=qC!�q                                    Bx\:��  
�          @�����@E� �����RCٚ����@S33��33�eC@                                     Bx\:�L  T          @�G���@�(���  �M�C
aH��@��׿�  �  C	z�                                    Bx\:��  �          @�(�����@����B�\��33C �3����@�33�#�
��{C �                                    Bx\:֘  T          @����G�@��R��Q��=G�C��G�@��H�c�
��\)CJ=                                    Bx\:�>  �          @�z�����@�  ��\)�S
=Cu�����@��Ϳ�\)���C
�                                    Bx\:��  
�          @��H����@����z��Qp�C�����@�zΐ33�p�C�R                                    Bx\;�  �          @�33��z�@����{����C�f��z�@��
��G��]C=q                                    Bx\;0  T          @����ff@���
=��(�CT{��ff@�ff����L(�C
=                                    Bx\;�  "          @�\���@�{�   �uG�C\���@�(����R�5�C
�3                                    Bx\;.|  T          @��H���@�
=�ff��z�C�����@�p���=q�@��C
�                                     Bx\;="  �          @���  @��ÿ�
=�6�\C	����  @���k����C�                                    Bx\;K�  
�          @�R����@���>��?�z�C�����@�33?@  @�
=C(�                                    Bx\;Zn  T          @�G��x��@���?�\)AB��R�x��@��
?޸RA^=qB�
=                                    Bx\;i  �          @����
@�{?��RA ��B��)���
@���?�=qAm��B�k�                                    Bx\;w�  T          @�Q��e�@�z�?��RA=G�B���e�@�ff@
=A��B�{                                    Bx\;�`  T          @�33��Q�@�
=��=q��CxR��Q�@�
=>�z�@�Cz�                                    Bx\;�  
�          @�p�����@�G���Q��6ffC=q����@���>W
=?�Q�C33                                    Bx\;��  �          @�z����
@��H��ff��
C�q���
@�p�������CJ=                                    Bx\;�R  
�          @ָR���\@^{� ����
=C�����\@i����33�l��CJ=                                    Bx\;��  T          @�Q����H@qG����
�(�C
���H@vff�&ff��ffCxR                                    Bx\;Ϟ  
Z          @���z�@�z��(��tz�C����z�@���#�
��C�{                                    Bx\;�D  �          @�p����@���(��(z�C	O\���@��ÿE����C�                                    Bx\;��  �          @ڏ\���@��R�p����(�C�)���@��þ���{�Cp�                                    Bx\;��  "          @�
=��p�@��׾B�\��\)C	ff��p�@���>�  @�C	k�                                    Bx\<
6  "          @�ff��(�@�
=?�@���C	�=��(�@���?uAQ�C
                                      Bx\<�  T          @������R@���>B�\?�\)Cu����R@�Q�?��@�(�C��                                    Bx\<'�  
Z          @��H����@��.{����CJ=����@�
=��=q��C��                                    Bx\<6(  T          @�  ��@xQ쾽p��Tz�C.��@x��    �#�
C�                                    Bx\<D�  �          @�����Q�@y���u�   C���Q�@x��>��R@5�C0�                                    Bx\<St  
�          @ȣ���33@������\)C
(���33@��>�=q@{C
5�                                    Bx\<b  �          @�
=��Q�@U?E�@�G�C�q��Q�@P��?�=qA��Cff                                    Bx\<p�  T          @љ���ff@fff>��H@�  C�=��ff@b�\?Q�@�p�C�q                                    Bx\<f  
�          @Ϯ���
@g�>\@X��C���
@dz�?8Q�@ʏ\Cc�                                    Bx\<�  �          @Ϯ���@~�R>B�\?�Q�C
���@|��?��@��CT{                                    Bx\<��  "          @�Q���@������z�CL���@��>�  @\)CW
                                    Bx\<�X  
�          @�{��{@�������c33CJ=��{@��
���uC&f                                    Bx\<��  �          @�Q���ff@x�þ����aG�C8R��ff@z=q�L�;�G�C{                                    Bx\<Ȥ  �          @ָR��Q�@����
=�c�
C����Q�@��L�;ǮC��                                    Bx\<�J  
�          @�Q���p�@��Ϳ
=���C�H��p�@�{�L�Ϳ���C��                                    Bx\<��  
�          @��
��  @�Q����L(�C�)��  @��ü���=qCz�                                    Bx\<��  T          @����33@�ff=L��>�p�Cs3��33@�>��@L��C�{                                    Bx\=<  T          @�p��Ǯ@~�R�8Q쿬��Cu��Ǯ@~�R>.{?���Cu�                                    Bx\=�  �          @�����@|�;����\)Cٚ����@|��>B�\?�  C޸                                   Bx\= �              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\=/.              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\==�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\=Lz              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\=[               ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\=i�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\=xl              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\=�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\=��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\=�^              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\=�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\=��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\=�P              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\=��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\=�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\=�B              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\>
�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\>�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\>(4              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\>6�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\>E�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\>T&              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\>b�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\>qr              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\>�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\>��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\>�d              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\>�
              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\>��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\>�V              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\>��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\>�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\>�H              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\?�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\?�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\?!:              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\?/�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\?>�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\?M,              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\?[�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\?jx              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\?y              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\?��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\?�j              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\?�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\?��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\?�\              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\?�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\?ߨ              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\?�N              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\?��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\@�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\@@              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\@(�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\@7�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\@F2              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\@T�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\@c~              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\@r$              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\@��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\@�p              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\@�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\@��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\@�b              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\@�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\@خ              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\@�T              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\@��   d          @�p���=q@Z�H?z�@�=qC\)��=q@XQ�?@  @��C��                                    Bx\A�  �          @�����@e�>aG�?�33C0�����@c�
>��@A�CL�                                    Bx\AF  "          @��
��33@Tz�>�\)@C&f��33@S33>�ff@XQ�CE                                    Bx\A!�  �          @�z���p�@N{>���@��C���p�@L��>��H@k�C0�                                    Bx\A0�  �          @�Q���z�@dz�<��
=�C�H��z�@c�
>L��?���C��                                    Bx\A?8  �          @����z�@aG���\)��\C���z�@aG�=�G�?L��C�                                    Bx\AM�  �          @�
=���
@`�׾L�Ϳ��HC����
@aG����
�#�
C��                                    Bx\A\�  
Z          @�\)�ָR@u��B�\����CL��ָR@u�    <�CE                                    Bx\Ak*  
�          @�  ��@x�þ�p��.�RCǮ��@z=q�8Q쿦ffC�3                                    Bx\Ay�  T          @�����(�@��þ�(��I��C�3��(�@����k���Q�C��                                    Bx\A�v  �          @�Q����@~{���aG�C8R���@\)��\)�C�                                    Bx\A�  T          @�  ��{@�����G��O\)CJ=��{@�녾k��ٙ�C33                                    Bx\A��  T          @�{����@r�\������CO\����@s�
��p��0  C(�                                    Bx\A�h  �          @����@hQ���y��Cff���@i����{�&ffCE                                    Bx\A�  �          @�z���p�@���k���33C� ��p�@��    <�CxR                                    Bx\AѴ  �          @��H��{@�녾�Q��'
=C!H��{@�=q���xQ�C�                                    Bx\A�Z  �          @�����Q�@��H�Ǯ�7�C����Q�@���8Q쿦ffC��                                    Bx\A�   �          @�(����\@��;\)���C����\@���=��
?(�C�                                    Bx\A��  �          @���z�@�=q�L�;�{C����z�@�=q>.{?���C�R                                    Bx\BL  "          @�����(�@��
=�G�?Y��Cff��(�@��>���@ ��Cs3                                    Bx\B�  
(          @��H�ȣ�@����\�p  Cc��ȣ�@������p�CG�                                    Bx\B)�  "          A���\)@�p��������C5���\)@�\)�����C�H                                    Bx\B8>  �          A�\����@�zῸQ��"�RC������@�ff��p��
=C8R                                    Bx\BF�  �          @�\)��{@�녿^�R��ffC����{@��H�+���ffCp�                                    Bx\BU�  �          @�  ��ff@�z�<�>aG�C0���ff@�(�>k�?�(�C8R                                    Bx\Bd0  "          @��H��ff@��\�aG��ָRCk���ff@���.{��{C5�                                    Bx\Br�  T          @�ff����@����33�E�C������@�\)�����-G�CJ=                                    Bx\B�|  T          @������@�G���Q��-��CxR����@��H���R�z�C!H                                    Bx\B�"  "          @����p�@��׿��R���C
s3��p�@�=q���
��C
.                                    Bx\B��  �          @�����@���B�\���C
����@��׿����{C	�                                    Bx\B�n              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\B�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\Bʺ            @陚���@�{�
=����C�)���@��R�����HQ�C�q                                   Bx\B�`  T          @�
=���\@�Q�=p���z�C����\@�G�������C޸                                   Bx\B�  �          @�  ��G�@��@  ��
=C���G�@��R�����  C�f                                   Bx\B��  �          A\)�Ϯ@���33�hQ�C��Ϯ@�(���\)�R�HCxR                                   Bx\CR  �          A z��˅@������Z�\C��˅@�녿��H�EG�CW
                                   Bx\C�  �          @����G�@�ff���R�/\)C����G�@�Q쿧��=qCc�                                   Bx\C"�  �          @���˅@�녿У��?33C޸�˅@��
���H�*�RC�                                     Bx\C1D  T          @��H��
=@�
=��\)� z�C���
=@��׿������C�)                                    Bx\C?�  �          @�\)��z�@�33�����!��CW
��z�@��Ϳ�Q��=qC                                    Bx\CN�  
�          @�(���ff@z=q��
�{�C��ff@\)��z��hQ�C@                                     Bx\C]6  �          AQ���(�@�  ���;
=CJ=��(�@�녿�G��(z�C�                                    Bx\Ck�  T          Ap����@�{�У��4(�C5����@����(��"{C�)                                    Bx\Cz�  �          A������@�33��ff�-�C�����@��Ϳ�33�\)Cn                                    Bx\C�(  "          Az���33@s33�)�����C����33@x��� ����33C\)                                    Bx\C��  
(          A��ָR@]p��1����C�3�ָR@dz��)�����C�                                    Bx\C�t  �          A ����(�@|(��ff��\)CG���(�@�������}G�C�                                     Bx\C�  
�          @�(����
@|(�?L��@�{C#����
@z=q?p��@�C\)                                    Bx\C��  
�          @�  ��z�@p  ?�  @�(�C�)��z�@mp�?���A�C޸                                    Bx\C�f  "          @�\)��Q�@��>�p�@7
=C���Q�@�G�?�@�  C(�                                    Bx\C�  "          @��R���H@��Ǯ�3�
C{���H@�{�k���33C�                                    Bx\C�  
�          @��R��{@��H�����C����{@�(��p����Q�CQ�                                    Bx\C�X  
�          @�p���z�@��R���$��C{��z�@�Q쿡G��{C�=                                    Bx\D�  T          @�{�\@�G���\�N�\C� �\@�33�����:=qCff                                    Bx\D�  
�          @�=q��ff@w��xQ���C�q��ff@y���W
=���CǮ                                    Bx\D*J  �          @����
@XQ쾔z��(�C����
@X�þ.{��ffC��                                    Bx\D8�  �          @�(���
=@AG��aG��У�C����
=@A녽�G��Tz�C}q                                    Bx\DG�  T          @�{���H@0��?5@��C
=���H@.�R?L��@�\)C8R                                    Bx\DV<  �          @��H�У�@Fff����RC���У�@HQ쿇���CW
                                    Bx\Dd�  
�          @����
@7���p��4Q�CO\���
@:�H�����((�C��                                    Bx\Ds�  �          @�����  @ff��Q��0��C!ff��  @����{�&�RC!�                                    Bx\D�.  �          @�ff�У�@.{�G����RCT{�У�@333������C                                    Bx\D��  �          @���z�@?\)�,�����\C\��z�@E��&ff��Cc�                                    Bx\D�z  
�          @��H��p�@���(���{C#���p�@!��ff���\C�                                    Bx\D�   T          @�����@Q��>�R��Q�C!����@�R�:=q��\)C �f                                    Bx\D��  
�          @�  ���H@  �<����p�C����H@ff�7���{C�H                                    Bx\D�l  �          @�ff��p�@(���QG�����C�3��p�@0  �K���  C                                    Bx\D�  �          @������?J=q�S�
��33C+�f����?c�
�Q�����C*�\                                    Bx\D�  "          @���Q�?u�<(���C*!H��Q�?�ff�:=q��
=C)+�                                    Bx\D�^  �          @�p��\�.{�'���ffC5�
�\���
�'����RC4�=                                    Bx\E  "          @�  ��(��L��������
C4xR��(�<��
������C3��                                    Bx\E�  
�          @�=q����>�{������RC0s3����>�
=�����C/��                                    Bx\E#P  T          @�G����\?(���	�����C-:����\?:�H�Q����
C,�=                                    Bx\E1�  
�          @�Q��Å>���L����\)C2�{�Å>8Q�L����C2W
                                    Bx\E@�  �          @Å���?���Z�H�
�HC#�����?�  �XQ����C"z�                                    Bx\EOB  �          @������?���mp���HC%G����?�G��j�H���C#�)                                    Bx\E]�  
�          @�G����?}p��A���G�C(�q���?�=q�@  ��ffC'��                                    Bx\El�  �          @�=q��=q?�p��>�R���C&\)��=q?����<(���ffC%aH                                    Bx\E{4  �          @��
��?�Q��N{� �C#��?���K����HC"�3                                    Bx\E��  �          @�{�xQ�@:�H�U�
=qC��xQ�@AG��P  ��C�                                    Bx\E��  �          @�  �Q�@�H���H�;p�C�{�Q�@#33�����7(�C)                                    Bx\E�&  T          @�  �^�R@"�\���\�-C���^�R@*�H��  �)�C��                                    Bx\E��  
Z          @�  ����@C33��33�4  C�����@E��ff�%�C�                                    Bx\E�r  "          @�p��Tz�?���{�D�C��Tz�@33��(��A�CT{                                    Bx\E�  �          @����W�@*�H�hQ�� �
C�
�W�@1��c33�z�CxR                                    Bx\E�  T          @�33�5�@��\@AG�A��RB�k��5�@\)@I��B  B��                                    Bx\E�d  �          @����c�
@��\@3�
A��HCff�c�
@��@;�A�p�C�R                                    Bx\E�
  �          @�z���\)@1녿�=q�%�C���\)@3�
�}p��(�C                                    Bx\F�  �          @�����p�?�(��C�
���HC&���p�?���AG����C%)                                    Bx\FV  "          @�����33�&ff�{��'
=C<xR��33���|���'��C:��                                    Bx\F*�  �          @�ff�\)?��R�8Q��
z�C"Ǯ�\)?����5�ffC!�                                    Bx\F9�  T          A�R��\)@�(��n{��=qC�H��\)@���e��  C�3                                    Bx\FHH  �          Az�����@���7
=���C	������@�{�,�����\C	L�                                    Bx\FV�  T          A{��
=@����Q���RC
��
=@�{��G��\)C�f                                    Bx\Fe�  �          A����
@�(���z����C	xR���
@�p���  ����C	L�                                    Bx\Ft:  
�          A Q�����@��
��=q�Tz�C(�����@���Q��C�C�)                                    Bx\F��  �          @�����=q@�Q��'�����Cff��=q@��H� ����\)C�f                                    Bx\F��  �          @�  ��=q@dz��p����C}q��=q@hQ��
=���RC�                                    Bx\F�,  �          @�\)��z�@e�\)��
=C���z�@i���Q���p�C�H                                    Bx\F��  �          @޸R���\@h�ÿ�z���=qC�f���\@l(����q�C�                                    Bx\F�x  �          @�  ��\)@K��4z����
C�H��\)@P���.�R��33CB�                                    Bx\F�  �          @أ����\@N�R�aG�����C8R���\@U��Z�H��=qCh�                                    Bx\F��  T          @ۅ���
@-p��|���=qC�����
@4z��w��
  C��                                    Bx\F�j  
�          @ٙ���
=@(Q��r�\��HC&f��
=@.�R�mp��C0�                                    Bx\F�  �          @�\)���@8���\(���\)Cn���@?\)�W
=��
=C�H                                    Bx\G�  �          @�Q���
=@Q���  �p�Cz���
=@\)�{���CxR                                    Bx\G\  T          @�z���Q�@����
��Cp���Q�@��������C=q                                    Bx\G$  �          @�=q���?��������  C#�R���?�(������C"��                                    Bx\G2�  �          @������?xQ��e�� �\C*+����?����c33����C)+�                                    Bx\GAN  �          @��H��G�?��H�Mp���(�C&u���G�?�ff�J�H��
=C%}q                                    Bx\GO�  "          @���33�L������C4p���33<#�
�����
C3�                                    Bx\G^�  �          @�����p�=��
������HC30���p�>���(�����C2�
                                    Bx\Gm@  T          @�=q������z���H����C6�f������  ���H����C6n                                    Bx\G{�  �          @\��z�>����,����(�C0�\��z�>\�,(���\)C0�                                    Bx\G��  �          @�z����H>�����
�-\)C-�
���H?z���33�,��C,T{                                    Bx\G�2  
Z          @�Q����;W
=��
=��33C6#����;#�
��Q����C5�
                                    Bx\G��  �          @����
=�#�
�������C4z���
=<��
�������C3Ǯ                                    Bx\G�~  T          @����{�?�����B  C+�{�?333��\)�A  C)�R                                    Bx\G�$  "          @����{=��\���z�C2�=��{>aG��\���=qC1\)                                    Bx\G��  
�          @�z���G���\)�1G����
C@����G�����333��\C?�R                                    Bx\G�p  �          @�G����׿xQ��+����HC>u����׿fff�-p����C=��                                    Bx\G�  �          @\��{�����  ��33CA�f��{���
��\��=qCAB�                                    Bx\G��  T          @�z�������ff��
=CH�������(��
=q���CH:�                                    Bx\Hb  �          @�Q���\)�W
=�Ǯ�h��CV���\)�S�
��33�w
=CU�f                                    Bx\H  �          @�(���p��˅��\)�m�CC\��p���ff���s�
CB�f                                    Bx\H+�  b          @�(����
=q�j�H��C9�R����G��l(��G�C8�
                                    Bx\H:T  x          @ə���G��Ǯ�l����C8ff��G���\)�mp��\)C78R                                    Bx\HH�  �          @�G���  ?����$z���ffC%5���  ?\�!���
=C$��                                    Bx\HW�  
�          @ʏ\���
?�
=�%��  C(+����
?�  �#33��G�C'u�                                    Bx\HfF  
�          @�  ��{?��������
=C&�)��{?�z��
=q��  C&�                                    Bx\Ht�  
�          @�p���(�?�\)�
=q����C(����(�?�
=�Q���Q�C(�                                    Bx\H��  
�          @����R?�p�������C"n���R?������C!�)                                    Bx\H�8  "          @�Q����?�
=��=q��G�C%n���?�p������  C$�                                    Bx\H��  
�          @�����Q�?��H��
=���RC'�f��Q�?�G�������C')                                    Bx\H��  T          @Å��Q�@7
=���H��z�CL���Q�@:=q������C�{                                    Bx\H�*  T          @��H��\)@��*�H��(�C�q��\)@���&ff����CO\                                    Bx\H��  "          @�Q���  @3�
�����
C����  @8Q��ff��\)C�                                    Bx\H�v  
�          @Å��33@<(���=q�p  C���33@>�R���R�b�HC�3                                    Bx\H�  
�          @Å��=q@[���G��@z�C�{��=q@]p���z��1G�CO\                                    Bx\H��  �          @����>�R@��þB�\��p�B왚�>�R@��ü���z�B�\                                    Bx\Ih  T          @��
�J�H@�������B�.�J�H@�p����
�A�B�\                                    Bx\I  
�          @�\)�)��@�p�����p�B�(��)��@�ff�k����B��                                    Bx\I$�  "          @�p��L(�@�  �33��B���L(�@���
=q���\B�u�                                    Bx\I3Z  T          @��H�]p�@���
=��Q�B�W
�]p�@�����(�����B��                                    Bx\IB   "          @�ff�A�@�z�����p�B���A�@�ff�Q����B��                                     Bx\IP�  
Z          @���?\)@�
=��p��U�B�ff�?\)@�Q쿨���=��B�
=                                    Bx\I_L  �          @�
=�z=q@�\)�Dz���\)B��\�z=q@�=q�:=q����B���                                    Bx\Im�  T          @�(���
@ƸR�^�R��B�����
@�\)�0������Bخ                                    Bx\I|�  
Z          @�(��:�H@���0����ffB�=�:�H@���'�����B�q                                    Bx\I�>  "          @�33�z=q������ff�F�CJ5��z=q������  �H�\CHY�                                    Bx\I��  "          @˅�~�R>�  ��{�LffC0u��~�R>Ǯ���K�HC.^�                                    Bx\I��  "          @�z��p  ?�Q���(��DffC��p  ?�=q���H�A�HC#�                                    Bx\I�0  �          @�ff�a�@Vff�Z=q�(�Cp��a�@\���S�
�p�C��                                    Bx\I��  �          @���k�@��R�   ��33C�=�k�@���������C
                                    Bx\I�|  
Z          @�G��U������\�q
=C4�
�U�>\)���\�p�C1�
                                    Bx\I�"  �          @���c33>8Q���{�l�\C1\�c33>�Q����l
=C.33                                    Bx\I��  "          @陚�xQ�?(���(��eQ�C+��xQ�?L���Å�d{C(k�                                    Bx\J n  "          @�R��33?�{���
�IC�3��33?��
��=q�G=qC�                                    Bx\J  "          @�\)�\(�@�Q쿮{�"ffB����\(�@ٙ���33�	��B�R                                    Bx\J�  �          @�z��j�H@�=q����f�\B�B��j�H@�(���
=�NffB��
                                    Bx\J,`  �          @������@aG����
�)�\C
������@j=q��Q��$�HC	�H                                    Bx\J;  
(          @��hQ�@��
�Z�H����B�Ǯ�hQ�@�
=�P����G�B��3                                    Bx\JI�  �          @��X��@��R�!G���ffB�aH�X��@�G��ff��=qB�R                                    Bx\JXR  �          @�R�S�
@�G��z����B� �S�
@�33����r{B�                                      Bx\Jf�  
�          @��L(�@ʏ\�����0��B�k��L(�@��
��Q��\)B��                                    Bx\Ju�  T          @�ff�Q�@�\)��p��=��B�\�Q�@ȣ׿��
�$Q�B�=q                                    Bx\J�D  
�          @�Q��S33@\�
=q���RB���S33@�z��(��|  B�k�                                    Bx\J��  �          @陚�j�H@�ff�z�H�B�
=�j�H@��\�p�����
B��R                                    Bx\J��  1          @��Vff@�G��^{���B��R�Vff@�z��S33�ۮB�                                    Bx\J�6  
Z          @���w�@�Q��K���Q�B�8R�w�@���@  ��Q�B�B�                                    Bx\J��  
�          @�{��  @�Q��Z�H����B�z���  @��
�P  ��
=B�aH                                    Bx\J͂  
�          @����33@�z�����\)C� ��33@�����\)�{C��                                    Bx\J�(  �          @���@�G���G��z�CJ=��@�p������	�Cc�                                    Bx\J��  
�          @����Q�@e��
=�!33C����Q�@o\)��33�\)C
�q                                    Bx\J�t  c          @���@J�H����+�Cs3��@U����R�'=qC{                                    Bx\K  E          @�Q�����@0����ff�/ffCJ=����@;���33�+p�CǮ                                    Bx\K�  "          @���Q�@4z���G��=�Cz���Q�@@����{�8�HC��                                    Bx\K%f  
�          @�R��ff@G�����6(�C:���ff@(���\)�2�RC��                                    Bx\K4  T          @�p����
@  ��ff�:\)C�����
@�H���
�6C�                                    Bx\KB�  
Z          @��r�\?�G����h�\C!�{�r�\?�p���z��e��C��                                    Bx\KQX  �          @��P  <#�
��
=u�C3Ǯ�P  >u��
=L�C/�=                                    Bx\K_�  
�          @�{�p  @�����=q���B���p  @�ff��z�� 33B�{                                    Bx\Kn�  
�          @���aG�@�z���{�&�B�\)�aG�@�=q����� ��B�W
                                    Bx\K}J  �          @����|(�@�ff��Q��Q�C�R�|(�@����33�=qC�                                    Bx\K��  	�          @�z��|(�@�G���(��)=qCT{�|(�@��R����#�C�                                    Bx\K��  	          @�  �j=q@G
=��  �Hz�C�)�j=q@S�
��z��C=qC	�)                                    Bx\K�<  �          A�R�u�@����� �B��3�u�@����=q�B��R                                    Bx\K��  
�          A�\�l(�@�{��Q��*C 33�l(�@�(����H�$\)B�33                                    Bx\Kƈ  "          Aff�j�H@\�z=q��\B�B��j�H@ƸR�l(��ظRB�(�                                    Bx\K�.  
�          A   �dz�@�����=q�\)B��dz�@�p����
�z�B�G�                                    Bx\K��  �          @���`��@�
=�q���
=B�#��`��@�33�dz���
=B���                                    Bx\K�z  
�          @���|��@��
�9����=qB���|��@�
=�,(����\B�
=                                    Bx\L   
�          @�z��|(�@�Q��E���B��
�|(�@���8Q���\)B��)                                    Bx\L�  
(          @��
���@��R�����HB�=q���@�G��	�����B��                                    Bx\Ll  "          @�\)�l(�@��p���Q�B�  �l(�@�Q��   ��(�B�L�                                    Bx\L-  �          @�������@��\�p���G�B�33����@�p��  ��\)B�k�                                    Bx\L;�  �          @��o\)@����@����\)B��o\)@�Q��2�\���B�                                    Bx\LJ^  	�          @�(��k�@����}p��{C Y��k�@�G��r�\� Q�B�                                      Bx\LY  
Z          @�ff�HQ�@��
�j�H��\B���HQ�@�Q��^{�噚B�L�                                    Bx\Lg�  T          @����B�\@��H�h�����RB�W
�B�\@�\)�[��噚B��                                    Bx\LvP  
Z          @�33�Dz�@�(��^�R���
B�#��Dz�@�Q��P����z�B�\                                    Bx\L��  "          @�G��P  @�Q��A���
=B��f�P  @��
�3�
��B���                                    Bx\L��  �          @�G��h��@���O\)��p�B���h��@����A��ĸRB�                                    Bx\L�B  
�          @��~�R@��H����z�B�W
�~�R@�����(�B�u�                                    Bx\L��  �          @�\)�z=q@������H�G\)B��z=q@����  �*�\B�33                                    Bx\L��  �          @�p��S�
@�������Yp�B�.�S�
@�G���  �;\)B���                                    Bx\L�4  
Z          @����U@��׿��\�G�B�\)�U@����Q���{B���                                    Bx\L��  
�          @\�\��@��R����G�B���\��@�\)��\)�*=qB��                                    Bx\L�  �          @\�Vff@�녾��R�:=qB�  �Vff@�=q���Ϳ}p�B��f                                    Bx\L�&  T          @�G��P��@��\�����\B�Q��P��@��\=�\)?.{B�Q�                                    Bx\M�  "          @�p��C�
@��?5@ҏ\B���C�
@���?k�A	p�B�=q                                    Bx\Mr  �          @Å�B�\@��?�@�33B�(��B�\@��R?G�@�(�B�k�                                    Bx\M&  �          @���Dz�@��ÿ}p����B����Dz�@�녿G����B�k�                                    Bx\M4�  �          @�{�C�
@�ff�����9�B�u��C�
@���}p��p�B���                                    Bx\MCd  
�          @�z��C33@����33��z�B���C33@�녿����d��B���                                    Bx\MR
  T          @�z��<(�@�Q����p�B�z��<(�@��\��{�}�B�R                                    Bx\M`�  �          @�=q�8Q�@������ŮB��H�8Q�@�ff�{��p�B��R                                    Bx\MoV  T          @����9��@n{�S33��HB���9��@vff�H����B��                                    Bx\M}�  "          @�z��AG�@n�R�Y���33C �q�AG�@w��N�R�z�B��H                                    Bx\M��  �          @�\)�:=q@Z�H�|(��%=qCh��:=q@e�r�\���C\                                    Bx\M�H  �          @�\)�6ff@z=q�`  �p�B�{�6ff@���Tz��G�B�\                                    Bx\M��  T          @�33��@�{�4z���(�B�{��@���&ff��B♚                                    Bx\M��  
�          @�33��p�@�{�8Q���
=B߸R��p�@���*=q��(�B�                                    Bx\M�:  
�          @�{�Q�@����8����33B�Ǯ�Q�@����*�H����B��                                    Bx\M��  �          @�33�S33@�Q��J=q��ffC� �S33@�z��>{��z�C �\                                    Bx\M�  	�          @�
=�,��@��R�7���z�B�z��,��@��\�*=q�ԸRB�{                                    Bx\M�,  �          @�녿��R@�{�����{B��f���R@�G���\���RB��                                    Bx\N�  �          @�Q��{@�z�����33Bݮ��{@���	�����B��f                                    Bx\Nx  �          @���>{@Z=q�l����C
=�>{@e��a���HC�3                                    Bx\N  "          @�z��:�H@h���b�\��C ���:�H@s33�W���\B�\                                    Bx\N-�  �          @��R�
=@�{��\)���B�W
�
=@��׿����B��                                    Bx\N<j  "          @��H��ff@�������F�RB�k���ff@��R�u��B�\                                    Bx\NK  
�          @���
=@�\)����Q�B�
=�
=@�����Q��qB�W
                                    Bx\NY�  T          @�G���R@�  ��
=�n�HB�B���R@�녿����H(�B��                                    Bx\Nh\  �          @�  ��\@�ff�Q��
=B�p���\@�\)����p�B�.                                    Bx\Nw  �          @��R�\)@�{�B�\���
B�Ǯ�\)@�
=�   ��=qB�=                                    Bx\N��  "          @��H���
@��=�G�?��B�����
@�33>�p�@u�B��H                                    Bx\N�N  �          @�G��n{@�33?�\A�z�Bǽq�n{@�Q�@�A���B�{                                    Bx\N��  �          @�G��L��@�
=@4z�A���BŅ�L��@�=q@Dz�A�33B�                                    Bx\N��  "          @�33��Q�@��H?�
=A��B��)��Q�@��@��A�B�aH                                    Bx\N�@  "          @�{���@�{?p��AB�uÿ��@�z�?�(�A>ffBϸR                                    Bx\N��  
�          @�{��(�@��?�Q�A9B�ff��(�@�G�?�(�Af�\Bѽq                                    Bx\N݌  �          @�33�K�@^{�|�����C���K�@j�H�p���{C�                                    Bx\N�2  
�          @љ��K�@\(������0�C��K�@j�H����'C�                                    Bx\N��  "          @���C�
@��������B��)�C�
@�33�s33�(�B�G�                                    Bx\O	~  �          @θR�XQ�@��P����z�B���XQ�@��H�AG���\)B��3                                    Bx\O$  1          @\�o\)@��Ϳ�\)�z�\Ck��o\)@�
=����U�C�                                    Bx\O&�  E          @�G��g�@�33�У����HCh��g�@���z��ep�C�)                                    Bx\O5p  
�          @�Q��p��@�����Q��MB�\)�p��@��
���&�RB���                                    Bx\OD  �          @ҏ\�vff@���%���33C�{�vff@�G�����(�C ��                                    Bx\OR�  
i          @�p��w
=@�G��B�\��Q�C �)�w
=@�{�1G�����B��                                    Bx\Oab  	�          @���2�\?�  ��33�d�C 8R�2�\?�  ��G��`=qC�
                                    Bx\Op  �          @��\�.�R?�녿�(���ffC!H�.�R?��H������HCٚ                                    Bx\O~�  �          @w��)��?�p�?�A��HCW
�)��?�\)?�z�A�CǮ                                    Bx\O�T  �          @Y���,��?��?   A33C8R�,��?�G�?
=A,Q�C��                                    Bx\O��  �          @*�H�aG��޸R�W
=����CsͿaG���Q�p���ѮCr^�                                    Bx\O��  
�          @�R�333��{�(�����RCy���333���ÿB�\����Cy�                                    Bx\O�F  
�          @{�5>��H�p��CB��5?�R�
�H�C�                                    Bx\O��  �          @X�ÿ\(�?�R�L(�L�C��\(�?L���I��#�C��                                    Bx\O֒  T          @�����>aG���Q���C(�����>�ff�\)�C=q                                    Bx\O�8  �          @���33<��
��ffB�C3O\��33>\��{z�C'\                                    Bx\O��  �          @Ϯ��Q��G���  ��Cb
��Q쿳33���HW
C[��                                    Bx\P�  "          @�G���������{ffC]����z�H��Q�p�CTY�                                    Bx\P*  "          @�(��޸R�W
=�˅G�C:�\�޸R>.{���
\)C.p�                                    Bx\P�            @��H������
=��p���Cn&f�����Ǯ�ȣ�=qChp�                                    Bx\P.v  �          @Ӆ��(������33L�Cd8R��(����\��{�C\�R                                    Bx\P=  �          @��� ��>B�\��(�
=C.�)� ��?
=q��33��C$�H                                    Bx\PK�  �          @��\����@��\�8�����
B�8R����@�  �&ff��
=B�=q                                    Bx\PZh  �          @�G���@�p���ff��=qB�k���@�  ��p��MB�                                    Bx\Pi  �          @�(��B�\@�\)?(��@�G�B�
=�B�\@�p�?�G�A ��B�.                                    Bx\Pw�  �          @�
=�\(�@�  ����!B��\(�@����.{���HB���                                    Bx\P�Z  �          @�
=����@��
���p�Bʣ׿���@�  �   ��G�B��                                    Bx\P�   �          @�\)�s33@����%����BȊ=�s33@�p������ffB���                                    Bx\P��  �          @�{��(�@��\������HBͳ3��(�@�ff��ff���\B��                                    Bx\P�L  �          @������@�(������9��B�(�����@�{�G�� Q�B��                                    Bx\P��  �          @��
����@�=q�}p��H(�B��Ϳ���@�(��5�
=B�z�                                    Bx\PϘ  �          @�G���  @(���)���$
=B�=q��  @3�
�{�{B�L�                                    Bx\P�>  �          @\)����@.�R�   ���B�\����@8���z���B�p�                                    Bx\P��  �          @z�H����@�R�,(��-��B�=q����@)���!G�� \)B�
=                                    Bx\P��  T          @�33��33@C�
�333�"\)B�
=��33@O\)�&ff��B���                                    Bx\Q
0  �          @��
���
����vffǮC<33���
=�G��vff�C.�                                    Bx\Q�  �          @��׿E��w
=���R�;��C�UÿE��dz����R�J��C���                                    Bx\Q'|  �          @���33�e���H�Sp�Cx�f��33�O\)��=q�a�RCv��                                    Bx\Q6"  �          @���\)�I����ff�hG�Cj��\)�.{�����t��Cfz�                                    Bx\QD�  �          @��H��
=�8����ff�v��ClE��
=�(���z���Cg��                                    Bx\QSn  �          @�ff����ff��
=Cb���Ϳ���ۅ=qC\��                                    Bx\Qb  �          @�p������/\)��
==qCtB������G���z�B�Co��                                    Bx\Qp�  �          @�����
�#33��G��3Co���
�z���ff\)Ci�{                                    Bx\Q`  �          @�\�W
=������{�AQ�C�Ǯ�W
=��p���  �Q�C�S3                                    Bx\Q�  �          @�zὣ�
��
=��33�7z�C�q콣�
�y����(��G�C�ff                                    Bx\Q��  �          @�p�>�p��]p����R�I�
C�>�p��I����ff�Z
=C�P�                                    Bx\Q�R  �          @����(��@  ��Q��oCF�R�(���������t(�C?�R                                    Bx\Q��  �          @����R��������x{Cg�����R�Ǯ����G�Cb8R                                    Bx\QȞ  �          @���>B�\�P���:�H�'(�C��{>B�\�B�\�I���7C��3                                    Bx\Q�D  �          @(��?.{��p��Ǯ�
=C�z�?.{��{�ٙ��"�
C��                                    Bx\Q��  �          @
=?�
=��Q�L����  C�]q?�
=���׿fff��C�                                      Bx\Q��  �          @p�?�G����>�ffA7�C�]q?�G���\)>�33A{C��                                    Bx\R6  �          ?޸R?�  �B�\>��
Ad(�C�c�?�  �G�>��A:ffC�                                      Bx\R�  �          @+�?�ff����?p��A��RC�C�?�ff����?W
=A�p�C���                                    Bx\R �  �          ?��ÿ�p����>�=qA$��C=޸��p���\)>�  Az�C>��                                    Bx\R/(  �          @��G�>\<��
?�\C)\)�G�>\=L��?��C)n                                    Bx\R=�  �          @
=q��p�?L��=�Q�@�RC�׿�p�?J=q>��@��
C�                                    Bx\RLt  �          @   ��ff?L��>#�
@�  C#׿�ff?G�>aG�@У�C�=                                    Bx\R[  �          @G���?^�R>ǮA!G�C�)��?W
=>�A<��C��                                    Bx\Ri�  �          @녿�z�?�ff>\A1p�C���z�?��\>��AW33C��                                    Bx\Rxf  �          @+��ff?�z�>�\)@�33C���ff?��>�p�Ap�C�                                    Bx\R�  �          @l(��HQ�?��?J=qAFffC=q�HQ�?�(�?n{Ai�C0�                                    Bx\R��  �          @��H�_\)?�
=?c�
AI��C��_\)?���?��AlQ�C�                                    Bx\R�X  �          @|(��Fff@ff?���A��C��Fff@   ?��A�(�C.                                    Bx\R��  T          @~�R�Z=q?�{?�=qA�Q�C�=�Z=q?�G�?��HA��C{                                    Bx\R��  �          @��H�{�?k�<�>�p�C&�=�{�?k�=���?�
=C&�)                                    Bx\R�J  �          @��
�p  ?��>��
@�z�C���p  ?�{>�G�@�  C !H                                    Bx\R��  �          @�{���H?Tz�=�G�?��C(� ���H?Q�>8Q�@{C(��                                    Bx\R�  �          @���33>���?�@��
C.aH��33>�Q�?\)@��C/�                                    Bx\R�<  �          @�ff����>\)?�RA ��C2&f����=�Q�?!G�A�RC2�{                                    Bx\S
�  �          @�����{>�?L��A-C20���{=u?O\)A/�C3)                                    Bx\S�  �          @����
==��
?��A[
=C2���
=�#�
?��A[�
C4�                                    Bx\S(.  �          @�{����#�
?Y��A'\)C4)������
?Y��A&�\C5�                                    Bx\S6�  �          @�z����H�u?
=q@�p�C7  ���H��z�?�@�z�C7�\                                    Bx\SEz  �          @�����þ����=q�UC7L����þu��z��g
=C7                                      Bx\ST   �          @��R��z��R��ff��C<����z�z�   ��z�C;��                                    Bx\Sb�  �          @z=q�l(���=q����(�CDQ��l(����
��R�(�CC�
                                    Bx\Sql  �          @�p���ff>\)��{�l��C2&f��ff>k�����iG�C0�
                                    Bx\S�  �          @��
���׾\�c�
�1p�C8� ���׾����k��7\)C7�q                                    Bx\S��  �          @�\)��>��H���
�[�C-W
��?z�z�H�QG�C,#�                                    Bx\S�^  �          @�Q��Z=q��p���ff��ffCR.�Z=q��Q��R�  CQ��                                    Bx\S�  �          @Tz��,�Ϳ���?z�HA��\CN��,�Ϳ�
=?\(�A{�CO��                                    Bx\S��  �          @Z�H���?�G����H��z�CY����?�{������
C�\                                    Bx\S�P  �          @�{��z�
=�Z�H� ��C:0���zᾊ=q�]p����C6��                                    Bx\S��  �          @��
�������8Q���
CA=q����E��>{�{C=��                                    Bx\S�  �          @�ff�����z���R��G�CI�R�����
=�)����=qCGxR                                    Bx\S�B  �          @�z�����G��ff��33CJ�=��녿����G�����CH��                                    Bx\T�  �          @����
=�
=�/\)����CN����
=���:�H�z�CK�{                                    Bx\T�  �          @��H�h�ÿ�(��_\)�#��CMT{�h�ÿ���h���,{CH�H                                    Bx\T!4  �          @��R�\���(��Mp���RCTaH�\�Ϳ�\)�Y���#�RCP��                                    Bx\T/�  T          @���8Q��=q�~�R�:��C[�)�8Q��G���{�G�
CW                                      Bx\T>�  �          @��G��P  ��(����HCb&f�G��Fff���
��{C`�q                                    Bx\TM&  �          @k����H�8�ÿ�
=���\CoY����H�0�׿�����=qCn5�                                    Bx\T[�  �          @o\)>#�
�333?h��BN=qC��>#�
�J=q?W
=B8��C��                                     Bx\Tjr  �          @�
=@�H?�\)@���B_��B\)@�H?���@�{Bj�
Aҏ\                                    Bx\Ty  �          @��H?Ǯ@g
=@c33B({B�G�?Ǯ@P  @xQ�B<z�B��3                                    Bx\T��  �          @�녾k�@���@<(�A�=qB��q�k�@�ff@^{B��B�                                    Bx\T�d  �          @�z�5@��H@\)A��B�uÿ5@��H@2�\Aڣ�B��                                    Bx\T�
  �          @ۅ?h��@�@1G�A�p�B�33?h��@�(�@W�A�z�B�W
                                    Bx\T��  �          @��
?@  @�\)@1�A���B�L�?@  @�p�@XQ�A�z�B���                                    Bx\T�V  �          @�=q?(��@Ǯ@%A���B���?(��@�ff@Mp�A�p�B�ff                                    Bx\T��  �          @޸R?�Q�@�  @��A�p�B�B�?�Q�@�  @2�\A��B�u�                                    Bx\Tߢ  �          @�\)?�=q@�z�@z�A�=qB�
=?�=q@�z�@,��A�
=B��)                                    Bx\T�H  T          @�
=?�(�@�ff?�p�Af�RB��?�(�@Ǯ@Q�A�=qB�                                    Bx\T��  �          @�\)?�
=@���?˅Ax��B�
=?�
=@��\@	��A�{B�
=                                    Bx\U�  �          @���n{@r�\@S�
B!33B�z�n{@[�@l(�B8=qB�B�                                    Bx\U:  �          @��Ϳ��@u�@4z�B�\B�Ǯ���@aG�@Mp�B$�B�ff                                    Bx\U(�  �          @�=q�Y��@��
@7�Bz�B���Y��@�G�@Tz�B  B��
                                    Bx\U7�  �          @�G�����@��\@(Q�A�(�Bڏ\����@���@FffB��B�\                                    Bx\UF,  T          @�G��W
=@���@�
A���B�
=�W
=@���@#33A�(�B�Q�                                    Bx\UT�  �          @�ff�8Q�@�G�@z�A�ffB�(��8Q�@�G�@%A�33B�aH                                    Bx\Ucx  �          @�ff�\@��?��A��B��\@�z�@33A�(�B�\)                                    Bx\Ur  �          @��
����@�p�?��RA�p�B�8R����@�@ ��A�RB��3                                    Bx\U��  �          @�p����@�G�?p��A(��B��쾅�@��?�p�A�  B���                                    Bx\U�j  �          @��=p�@�G��\)��B��)�=p�@�=q<��
>��B�                                    Bx\U�  �          @���>#�
@��@0��A���B�
=>#�
@�=q@Z�HA�33B��)                                    Bx\U��  �          @�G��.{@�(�?�
=AAG�B��R�.{@�@	��A���B�{                                    Bx\U�\  �          @�(�?&ff@Å@{A�33B��)?&ff@���@HQ�A�B�33                                    Bx\U�  �          @��>�@�(�@ffA���B���>�@\@B�\AԸRB��\                                    Bx\Uب  �          @�p�����@�\)@��A�G�B��H����@�{@<(�AҸRB���                                    Bx\U�N  �          @�녽��
@˅@��A�\)B�ff���
@���@FffA�
=B�z�                                    Bx\U��  �          @��
�#�
@�\)@G�A�=qB�=q�#�
@�@?\)A�=qB�=q                                    Bx\V�  �          @�(�>�p�@�@�A��
B�(�>�p�@�(�@>{A�  B���                                    Bx\V@  �          @�Q�>��R@��@�RA���B�#�>��R@�p�@I��A�{B���                                    Bx\V!�  �          @��H?Q�@�?��AffB�aH?Q�@ȣ�?��
A{33B�                                    Bx\V0�  �          @љ�>�p�@�\)?G�@�=qB�ff>�p�@�33?\AW33B�B�                                    Bx\V?2  T          @ȣ�>8Q�@��H?�z�AP��B���>8Q�@�(�@
=A��
B�z�                                    Bx\VM�  �          @���<#�
@�p�?�{A*=qB�<#�
@�  ?�ffA��HB�                                    Bx\V\~  �          @Å=�\)@���?G�@��
B��R=�\)@�p�?��RAaB��3                                    Bx\Vk$  �          @�G��\)@�  ?.{@�
=B����\)@�(�?���AT(�B��R                                    Bx\Vy�  �          @�ff��@��?�33A2�RB�=q��@�(�?�=qA�B��\                                    Bx\V�p  �          @��=p�@�G�?��
A   B��=p�@�(�?��HA�ffB�
=                                    Bx\V�  T          @��J=q@�=q?J=q@��
B�ff�J=q@�?�p�Ag33B�                                    Bx\V��  �          @�{�\(�@�=q?Q�@�BĽq�\(�@�?\Alz�B�(�                                    Bx\V�b  �          @��׿k�@���?:�H@߮BŸR�k�@���?�Q�A]�B��                                    Bx\V�  �          @�G���=q@�z�?J=q@�G�BȽq��=q@�  ?�  Af�HB�=q                                    Bx\VѮ  �          @��R�}p�@���?p��A��B�ff�}p�@�z�?��A�
B��                                    Bx\V�T  �          @��׿�G�@�33?xQ�A��BǏ\��G�@�{?�Q�A�(�B��                                    Bx\V��  �          @�33��{@�?h��A��B�#׿�{@���?��Ax��BɸR                                    Bx\V��  �          @�
=��z�@���?xQ�A{B��쿔z�@�33?�
=A�
=B�z�                                    Bx\WF  �          @��ÿp��@�(�?h��A\)B�\�p��@�
=?��A|��BƏ\                                    Bx\W�  �          @�33�xQ�@�ff?n{AQ�B�p��xQ�@�G�?�A~ffB��                                    Bx\W)�  �          @��u@�=q?(�@�{B��Ϳu@�{?��AQ��B�33                                    Bx\W88  �          @�녿u@�
=?�@��
B�8R�u@�33?��\AAG�BƔ{                                    Bx\WF�  �          @��H��  @��?�{AF{B�B���  @��\@�A�z�B��                                    Bx\WU�  �          @�G��O\)@�=q?���A1�B�33�O\)@��?�(�A�p�Býq                                    Bx\Wd*  �          @�
=�=p�@�{?˅Ae�B��3�=p�@�{@Q�A�p�B�L�                                    Bx\Wr�  �          @�=q�J=q@��
?�=qAB{B��J=q@���@
=A�{B�.                                    Bx\W�v  �          @�Q�(�@���?p��A��B���(�@�
=?�G�A|  B��                                    Bx\W�  �          @Ϯ��@��?0��@��B��=��@�Q�?��
AZ�\B�                                    Bx\W��  �          @љ��k�@�{?
=q@�
=B�8R�k�@�=q?���AC33BĊ=                                    Bx\W�h  �          @�  �.{@�ff>\)?��
B�\�.{@�(�?�  Ap�B�8R                                    Bx\W�  T          @�녿\)@У׾8Q�ǮB��
�\)@Ϯ?0��@���B��H                                    Bx\Wʴ  �          @׮��=q@�z�>�Q�@E�BƊ=��=q@���?�  A*ffB��
                                    Bx\W�Z  �          @�p���ff@ڏ\=�\)?(�Bŀ ��ff@�Q�?}p�A�BŮ                                    Bx\W�   �          @��H�h��@�Q�>8Q�?�  B�G��h��@�?��A
=B�z�                                    Bx\W��  �          @�  �B�\@�{�L�Ϳ޸RB�� �B�\@���?.{@�  B��{                                    Bx\XL  �          @���@�ff��{� (�B�\)��@�G������
B�
=                                    Bx\X�  �          @�33����@�
=?s33A�HB�������@�G�?�p�A�Q�B��
                                    Bx\X"�  �          @�녾��
@���>�G�@��B�
=���
@���?�G�A@��B�(�                                    Bx\X1>  �          @�=q��p�@���?(�@��B�  ��p�@�(�?�Q�A[�
B�(�                                    Bx\X?�  �          @��ÿ
=q@ƸR?E�@��B��f�
=q@�G�?У�ApQ�B�(�                                    Bx\XN�  �          @�ff�.{@�(�?(�@�p�B�8R�.{@Ǯ?��RAVffB��                                     Bx\X]0  �          @�=q�(��@ȣ�>u@p�B�  �(��@�p�?�\)A#�B�.                                    Bx\Xk�  �          @��H�
=@ə�������B�Ǯ�
=@�Q�?E�@޸RB��)                                    Bx\Xz|  T          @�(����
@�33��{�W
=B�W
���
@��H>��H@���B�\)                                    Bx\X�"  �          @���#�
@�{<��
>k�B�Q�#�
@�(�?^�RAB�u�                                    Bx\X��  �          @��׿&ff@�\)>��?�B�k��&ff@���?�  Ap�B��{                                    Bx\X�n  �          @�Q�
=q@�
==�Q�?Y��B�ff�
=q@���?s33AG�B��                                    Bx\X�  �          @��׾Ǯ@���#�
��
=B�� �Ǯ@�{?Q�@��B��\                                    Bx\Xú  �          @�(���\)@��
    <#�
B�L;�\)@���?c�
AG�B�\)                                    Bx\X�`  �          @Å���@�33=#�
>��B�Ǯ���@���?n{A(�B��H                                    Bx\X�  �          @ƸR��p�@�{��G���  B�𤾽p�@�z�?L��@���B�                                      Bx\X�  �          @���u@��
�Ǯ�k�B�(��u@Å?�\@���B�(�                                    Bx\X�R  �          @�
=��G�@�
=<��
>aG�B����G�@���?p��A\)B�                                    Bx\Y�  �          @ƸR��@�ff>k�@�B�Q��@��H?��A)�B�\)                                    Bx\Y�  �          @�\)�#�
@�
=>aG�@z�B���#�
@��
?�33A*=qB��                                    Bx\Y*D  �          @Å�#�
@�33>��
@C33B�
=�#�
@�\)?�p�A:ffB��                                    Bx\Y8�  �          @��H��\)@�=q>�33@P  B�W
��\)@�ff?�  A>=qB�u�                                    Bx\YG�  �          @�녿\)@ȣ�>Ǯ@aG�B���\)@�z�?�=qAB�HB�W
                                    Bx\YV6  �          @�33�Q�@�Q�?#�
@��B�aH�Q�@��H?У�AdQ�B�                                    Bx\Yd�  �          @�33��R@���?z�@���B��f��R@˅?���A\z�B�.                                    Bx\Ys�  T          @��
�}p�@�G�>8Q�?�ffB�B��}p�@�{?�
=A$(�BŊ=                                    Bx\Y�(  �          @��H>��H@љ�>aG�?�Q�B�aH>��H@�{?�p�A,(�B�=q                                    Bx\Y��  
�          @�33?   @���>��?�ffB�L�?   @�?�z�A"�RB�(�                                    Bx\Y�t  �          @��ÿJ=q@�ff?�@��B�
=�J=q@���?ǮA]��B�ff                                    Bx\Y�  �          @љ�>W
=@�  ?333@�(�B�Q�>W
=@�=q?ٙ�Aq��B�33                                    Bx\Y��  �          @�
=���
@�?�@���B������
@�Q�?���A`Q�B���                                    Bx\Y�f  �          @��H�n{@�  >.{?�G�B���n{@���?�33A'\)B�B�                                    Bx\Y�  �          @�G���@�(����
�#�
B����@���?s33A33B�ff                                    Bx\Y�  T          @��ÿ�Q�@���#�
��B��
��Q�@�33?Q�@�G�B�
=                                    Bx\Y�X  �          @�{��G�@�������p�B�zῡG�@�  ?Tz�@�  B˳3                                    Bx\Z�  �          @�Q쿜(�@�33��\���RBʔ{��(�@Å>��@��\Bʏ\                                    Bx\Z�  T          @�\)��G�@�33=���?h��Bʅ��G�@�  ?�\)A�B��)                                    Bx\Z#J  T          @љ��O\)@�?h��@�\)B�\)�O\)@�ff?�Q�A���B��f                                    Bx\Z1�  �          @�ff�p��@ə�?h��A{B��H�p��@�=q?�A��HBŅ                                    Bx\Z@�  �          @�{�h��@ʏ\?�@��
B�LͿh��@���?���Af=qB�                                    Bx\ZO<  T          @��
�L��@�G�>�ff@�G�B\�L��@�(�?�(�AU�B��                                    Bx\Z]�  �          @�p����@�z�>W
=?�B�Q���@ȣ�?�  A4Q�B�z�                                    Bx\Zl�  �          @��H�
=q@��=��
?0��B�ff�
=q@θR?�z�A"ffB��\                                    Bx\Z{.  �          @�(�����@��
    <#�
B������@���?���Az�B�33                                    Bx\Z��  �          @�G��W
=@θR��\)�&ffB��ͿW
=@�(�?�  Ap�B���                                    Bx\Z�z  �          @�
=���@��H�\(��Q�Bڽq���@�p�<#�
=L��B�=q                                    Bx\Z�   T          @�z��\)@�{�\)����B�aH�\)@�ff>�ff@�Q�B�Q�                                    Bx\Z��  �          @�z��G�@�\)�h�����B��f�G�@���=u?(�B�u�                                    Bx\Z�l  �          @�33�ff@��Ϳh�����B��H�ff@�
==u?z�B�ff                                    Bx\Z�  �          @�Q�� ��@�Q쿣�
�E�B�33� ��@�zᾨ���G�B�L�                                    Bx\Z�  �          @\��\@�ff���\��Bօ��\@�G��#�
��33B�                                    Bx\Z�^  �          @���Q�@�>�@�z�B�z��Q�@�Q�?���AZ�\B�G�                                    Bx\Z�  �          @�  ��@�{?L��@ڏ\B���@ƸR?�33A��RB�{                                   Bx\[�  �          @�녿Ǯ@У�?�A{B��)�Ǯ@�
=@G�A�=qB��                                   Bx\[P  �          @׮����@�?��A1B�k�����@Å@��A���BθR                                    Bx\[*�  �          @��H���@�(�?���A��B�����@\@�RA�Q�B�Ǯ                                    Bx\[9�  �          @��H��@��ÿ�p��W�
B�=q��@�ff��G�����B�.                                    Bx\[HB  �          @Ӆ�@ƸR�=p���B�.�@Ǯ>�Q�@I��B�                                      Bx\[V�  �          @��ÿ���@Ǯ>Ǯ@\(�Bԏ\����@�=q?�p�AR�RB�ff                                    Bx\[e�  T          @���ff@�G�?��RA4(�B�#��ff@�\)@  A�G�B�ff                                    Bx\[t4  �          @���\)@�G�?���A(Q�Bܮ�\)@�\)@  A���Bޮ                                    Bx\[��  �          @�(���Q�@�33?���Ad��Bب���Q�@�\)@%A�{B���                                    Bx\[��  �          @�ff���@���>�\)@"�\B����@�Q�?�=qA@��B�
=                                    Bx\[�&  �          @�z����@�{��
=�2�RBׅ����@�녾\)���B��
                                    Bx\[��  �          @��H�Q�@���s33�z�Bݏ\�Q�@�{=�\)?0��B�
=                                    Bx\[�r  �          @���(�@����
�x��B虚�(�@�(��(����B��
                                    Bx\[�  �          @��
��@������;\)B�
=��@���aG����B���                                    Bx\[ھ  �          @�(���R@�녿�  �$��B㞸��R@��ͽu�&ffB���                                    Bx\[�d  �          @��\��@�=q��Q��p��B�\)��@�  �
=q����B�                                    Bx\[�
  �          @�(���@�
=���?�B�����@�33�k��Q�B��)                                    Bx\\�  T          @�{���@����=q���\B�
=���@�p��c�
�z�B���                                    Bx\\V  �          @�z��(�@������B�=q��(�@��R�����T��B�
=                                    Bx\\#�  �          @�����@����p���  B۞���@�33���R�F{B�aH                                    Bx\\2�  �          @���
=@��
�
�H���B�G���
=@�{��(��F�RB��
                                    Bx\\AH  �          @��
��@��R�8�����
B�.��@�� ����G�B�aH                                    Bx\\O�  �          @����@��;���33B���@��Ϳ��R���B�R                                    Bx\\^�  �          @���@���Mp���HB����@�����
��  B�#�                                    Bx\\m:  �          @�녿�Q�@��H�h���(�B��
��Q�@�ff�333��ffB��H                                    Bx\\{�  �          @��ÿ˅@�\)�e�B���˅@��H�.{��RB�W
                                    Bx\\��  �          @�����  @��
�`����HB��
��  @��R�'
=���Bՙ�                                    Bx\\�,  �          @�p���z�@��\�e��RB�{��z�@��,(���B�G�                                    Bx\\��  �          @�����@���n�R�Q�B�
=���@�ff�8Q���33B��f                                    Bx\\�x  �          @�(���p�@Dz���  �EQ�B�z��p�@u��e���B�                                     Bx\\�  T          @�
=�p�@U����5��B��p�@���U��=qB��                                    Bx\\��  
�          @��
����@����p��h�C ����@G������C��B�{                                    Bx\\�j  �          @�(��33@E��n{�0\)B�u��33@o\)�C33�  B�\                                    Bx\\�  �          @�p���@�  ?�A��B�����@�=q@*�HA�33B�z�                                    Bx\\��  �          @�G���@��H?��A�(�Bޙ���@��@=p�A�=qB�8R                                    Bx\]\  �          @���@��@�A��B�.�@���@K�A��
B�{                                    Bx\]  �          @�(����@��?��A��RBߊ=���@�ff@7�A�
=B�
=                                    Bx\]+�  �          @�=q��H@��?��A�B�
=��H@��H@7
=A�G�B���                                    Bx\]:N  �          @�G��(Q�@�Q�?�(�AY�B���(Q�@��@"�\A��B�ff                                    Bx\]H�  �          @�Q��333@�z�?���AV�HB����333@��@   A��RB��                                    Bx\]W�  �          @�
=�'
=@��?fffA�B�ff�'
=@�Q�@   A�z�B��
                                    Bx\]f@  �          @����;�@�Q�?�R@��B��;�@���?�(�A~{B�#�                                    Bx\]t�  �          @ə��Dz�@�\)?�@��HB�z��Dz�@�Q�?У�Ao�B��                                    Bx\]��  
�          @�=q�G�@�\)>\@Z=qB�Q��G�@�G�?�p�AX��B�{                                    Bx\]�2  �          @���S33@�p�>�@�  B�\�S33@�
=?��
AeB�33                                    Bx\]��  �          @����?\)@���?Y��@�
=B�q�?\)@���?�A��B�                                     Bx\]�~  �          @�{�c33@���>�@�G�B�p��c33@��\?�p�A^ffB���                                    Bx\]�$  �          @�{�Q�@�p�?s33A��B����Q�@��?��RA��B�                                      Bx\]��  �          @�
=�7
=@��\?��AG\)B�p��7
=@�{@=qA���B�(�                                    Bx\]�p  �          @��
�   @���?�AW33B噚�   @��@ ��A��B�#�                                    Bx\]�  �          @ƸR�8Q�@��
?�  A9�B��8Q�@��@A��HB�{                                    Bx\]��  �          @ʏ\�O\)@��?\(�@��\B��O\)@���?�Q�A�Q�B��R                                    Bx\^b  �          @���
=@�p�@
�HA�33B����
=@�33@P��A�
=B��
                                    Bx\^  �          @�{��{@�\)@3�
A��B�k���{@���@y��B(�Bފ=                                    Bx\^$�  �          @˅�{@��@
=A�(�B�=�{@��@[�B��B�{                                    Bx\^3T  �          @���Z=q@���?���A@z�B�
=�Z=q@�Q�@Q�A���B�Q�                                    Bx\^A�  �          @����^{@���?h��A�\B��q�^{@��R@   A�z�B���                                    Bx\^P�  �          @θR�s33@�p����
�0��B��q�s33@��?��\A�HB��
                                    Bx\^_F  �          @�G��w�@��R��R��B�B��w�@�
=?�\@�\)B�#�                                    Bx\^m�  �          @���`��@���=���?fffB�\)�`��@��
?�p�A2�\B���                                    Bx\^|�  �          @��\@�
=@p�A�ffBѳ3�\@���@hQ�B
  B�aH                                    Bx\^�8  �          @θR� ��@��H@\)A�  B۔{� ��@�p�@h��B	�B�u�                                    Bx\^��  �          @�  �*=q@���@�\A�G�B�\)�*=q@��R@L(�A�z�B�k�                                    Bx\^��  
�          @�\)�_\)@��
?���A���B�p��_\)@��@8��A�z�B�8R                                    Bx\^�*  �          @�{�K�@��
@
=A��B�\)�K�@�\)@W�B��B��
                                    Bx\^��  �          @�
=���
@qG�@Q�A�G�C	�)���
@=p�@���B�\CY�                                    Bx\^�v  �          @�����@qG�@G�A���C	������@@  @w�Bp�C&f                                    Bx\^�  �          @��
��33@XQ�@J�HA�
=C0���33@&ff@u�BG�C(�                                    Bx\^��  �          @����z�H@��@�A�C\�z�H@���@P  AC�                                    Bx\_ h  �          @�=q���@�p�@  A�C  ���@���@N{A�RC                                    Bx\_  �          @����{�@�z�@�A�  C=q�{�@�  @UA�  Cz�                                    Bx\_�  �          @�
=���R@�  @{A��C� ���R@e@W
=A��C��                                    Bx\_,Z  �          @�G���p�@��@{A�z�C� ��p�@l(�@X��A�Q�C
xR                                    Bx\_;   �          @��H��z�@��\@�
A��
C
�f��z�@]p�@J=qA�G�CE                                    Bx\_I�  �          @����
=@dz�@�A��\CO\��
=@=p�@Dz�A߮CB�                                    Bx\_XL  �          @�����Q�@��
@z�A�\)C���Q�@c33@<(�A��HCL�                                    Bx\_f�  �          @љ���z�@=p�@H��A�=qC�)��z�@
�H@n�RBffC
                                    Bx\_u�  �          @�=q��=q@��@S�
A�CB���=q?У�@q�Bp�C".                                    Bx\_�>  T          @�����{@@��@E�A�ffC�f��{@�R@l(�B�HC��                                    Bx\_��  �          @�  ���H@U@3�
A�=qCǮ���H@'
=@`  A��C�f                                    Bx\_��  �          @�
=��  @��\@�HA���CaH��  @Z=q@R�\A��CO\                                    Bx\_�0  T          @أ���ff@j=q@L(�A�33C���ff@5�@|(�B�
C�f                                    Bx\_��  
�          @ٙ����@c33@tz�B�RC�����@%�@���B'\)Ch�                                    Bx\_�|  �          @�(���ff@o\)@j=qB ��C�R��ff@333@�p�B =qC�H                                    Bx\_�"  �          @ٙ����
@p��@c33A�
=CJ=���
@5@��\B��C�                                    Bx\_��  �          @أ���
=@~�R@1G�A��C�)��
=@N{@g�B�C�f                                    Bx\_�n  �          @�Q����H@��@1�A�  C
!H���H@Vff@j=qB{Cٚ                                    Bx\`  �          @׮��=q@���@A��C���=q@X��@N{A�p�C�
                                    Bx\`�  �          @�Q�����@���@>{A�ffC
�\����@Mp�@u�B
  C��                                    Bx\`%`  �          @����{@���@2�\A���C5���{@Q�@j=qBQ�C
=                                    Bx\`4  �          @ڏ\��\)@�p�@(��A��C
�)��\)@[�@c33A�Q�C\                                    Bx\`B�  T          @�33����@�ff@   A��
Cz�����@o\)@^�RA�\)Cz�                                    Bx\`QR  �          @�z���ff@��@�A�{C
��ff@��\@S�
A�C#�                                    Bx\`_�  T          @أ���G�@�ff@(��A���C	@ ��G�@\��@c�
A��
CǮ                                    Bx\`n�  
�          @׮��  @`��@6ffA�=qC���  @.�R@g
=BffCaH                                    Bx\`}D  
�          @أ���{@fff?�p�A�p�C=q��{@AG�@2�\A�33CǮ                                    Bx\`��  �          @�=q��Q�@�z�@
=A�Q�C	J=��Q�@p��@FffAظRC�3                                    Bx\`��  �          @׮���\@�33@z�A��CxR���\@j=q@R�\A�\CT{                                    Bx\`�6  �          @�=q���@�33@�A�33C	����@hQ�@Z=qA��C
                                    Bx\`��  �          @��H���@�Q�@(��A�p�C	�����@`  @eA�G�C
                                    Bx\`Ƃ  �          @ڏ\��{@�G�@   A�(�C	���{@c33@^{A�ffC��                                    Bx\`�(  �          @ٙ���
=@��\@)��A��C&f��
=@Tz�@c�
A��C�                                    Bx\`��  �          @�
=���\@u�@(Q�A���C�����\@Dz�@_\)A���C�                                    Bx\`�t  �          @�\)��G�@qG�@ffA��
C+���G�@E�@L��A�C��                                    Bx\a  T          @أ����@z�H@�
A�C�=���@R�\@=p�A�p�C8R                                    Bx\a�  �          @أ�����@qG�?�Q�A�p�Cc�����@K�@3�
A�ffC��                                    Bx\af  �          @����z�@�
=?�z�A>�HC�
��z�@o\)@��A��C�                                    Bx\a-  �          @������@�
=?uA��CW
����@��H@A��
C	                                    Bx\a;�  T          @أ���@�  ?�{A33C	����@��\@(�A�33Ch�                                    Bx\aJX  �          @�  ��33@���?�  A+33C	��33@��@A�(�C                                    Bx\aX�  �          @׮����@�33?z�HAC)����@~{@�A��RC��                                    Bx\ag�  �          @�ff���@~�R?�G�AtQ�C�\���@Z�H@,��A��RC�q                                    Bx\avJ  �          @�p���=q@dz�?�\)Ac�C&f��=q@C33@{A�\)C33                                    Bx\a��  �          @��H����@n{?�{A>�RC������@P��@  A�  C8R                                    Bx\a��  �          @�33���@k�?��AQ�C�����@R�\?�p�A�{CxR                                    Bx\a�<  �          @�(���G�@��>Ǯ@W
=Cff��G�@��?��
AUp�C	��                                    Bx\a��  �          @����(�@�33>�(�@s33C
��(�@��\?�33AiG�C�R                                    Bx\a��  �          @љ���\)@�z�?n{A=qCJ=��\)@p��?���A�z�C��                                    Bx\a�.  �          @������@��
?�
=AJ{C����@w
=@   A��CG�                                    Bx\a��  �          @�����z�@��?��HAr�HC	�
��z�@j=q@.�RA�ffC�R                                    Bx\a�z  T          @Ϯ���
@�33?�\)AB�\C�����
@�33@   A��C޸                                    Bx\a�   �          @����Q�@��H?�z�A(��C޸��Q�@�(�@33A��Cٚ                                    Bx\b�  �          @�z���(�@�33?˅Ag\)C����(�@���@.{A�(�C��                                    Bx\bl  �          @�z��fff@��R?��HAy�B����fff@�33@;�A�C�H                                    Bx\b&  �          @ʏ\�W
=@���?�{A��HB���W
=@��
@FffA�\B�\                                    Bx\b4�  �          @ʏ\�5@��?ǮAeB�k��5@�=q@:=qA�(�B�\                                    Bx\bC^  �          @�p���p�@j�H@\)A���CO\��p�@=p�@G�A��C33                                    Bx\bR  �          @�z����@���?��A��C�3���@`��@:=qA�(�Cs3                                    Bx\b`�  �          @�z��|(�@��H?���A��C@ �|(�@n{@8��A�z�C�3                                    Bx\boP  �          @�p��p  @���?��A�33C�=�p  @z�H@:�HA�Q�C                                    Bx\b}�  �          @���l��@�=q?�G�A��C ���l��@|��@9��AᙚC#�                                    Bx\b��  
�          @�����@��?ٙ�A�ffC������@p  @2�\A�\)C�                                    Bx\b�B  T          @ƸR��  @��\?�
=A��C!H��  @[�@<(�A�z�C
                                    Bx\b��  �          @�ff��  @�33?��A�{C  ��  @\��@:=qA��
C޸                                    Bx\b��  �          @ʏ\��  @�  ?���A��C�)��  @tz�@E�A��CJ=                                    Bx\b�4  �          @��H���@���?�Q�Ax  CQ����@y��@5�A�z�C�                                     Bx\b��  �          @�33���@�z�@G�A���C����@l(�@G�A�C
�                                    Bx\b�  �          @�(����
@��@33A�=qC����
@l��@J=qA�p�C
\                                    Bx\b�&  �          @�p��\)@���?�z�A��RC���\)@~{@EA�(�C&f                                    Bx\c�  �          @�����@�  ?���A�G�C�����@s�
@E�A�C	n                                    Bx\cr  �          @�
=���@���?��A�Q�CB����@j=q@8Q�A�\)C�{                                    Bx\c  �          @�G���ff@|��?�=qA_�CaH��ff@XQ�@%�A��\C��                                    Bx\c-�  �          @�����G�@��?��HA,  C
� ��G�@p  @�
A���C�                                    Bx\c<d  �          @�33���\@���?���AG�Cn���\@g
=@�A���C��                                    Bx\cK
  T          @Ϯ��p�@��H@��A�C����p�@aG�@_\)B33C��                                    Bx\cY�  �          @�  ����@�z�@!�A���C�����@b�\@hQ�B{C
�)                                    Bx\chV  �          @�p����@�p�?��RA�(�C+����@mp�@G�A�=qC!H                                    Bx\cv�  �          @�
=��G�@���@��A�33C����G�@g
=@W�A�p�Cz�                                    Bx\c��  �          @ٙ��hQ�@��\@'�A��
B���hQ�@�p�@z=qB{C\                                    Bx\c�H  �          @ٙ��g
=@�
=@^�RA��C ���g
=@U@��\B+�C	+�                                    Bx\c��  �          @����,(�@�p�@H��A���B����,(�@�33@�{B(B��                                     Bx\c��  �          @����s�
@��H@<(�A�  C���s�
@e@��HB��C��                                    Bx\c�:  �          @�����\)@xQ�@>{A��HC)��\)@:=q@{�BG�C�                                    Bx\c��  �          @׮��33@Z�H@1�A�33C+���33@ ��@g�B
=CǮ                                    Bx\c݆  �          @���33@7�@7
=AɅC� ��33?�(�@b�\A��HC�\                                    Bx\c�,  �          @�������@.{@5�A��C�����?���@^{B��C !H                                    Bx\c��  �          @ȣ���=q@�@$z�A�C����=q?��
@C33A���C&z�                                    Bx\d	x  �          @��H���\@�@
�HA��\C   ���\?��@)��A�Q�C&xR                                    Bx\d  �          @�{����@�?�z�Ax��C������?У�@��A���C#��                                    Bx\d&�  �          @�p���
=@
=?�33A��C޸��
=?�  @�HA���C$�R                                    Bx\d5j  �          @ƸR���
?�z�?�A�\)C!@ ���
?��@33A��C&�
                                    Bx\dD  �          @����@�?޸RA�(�C�\��?��@G�A��\C$�
                                    Bx\dR�  �          @����z�@�?�(�A�\)C B���z�?��@p�A���C&33                                    Bx\da\  �          @�����
=@�
@Q�A���C)��
=?��@<(�A��HC$J=                                    Bx\dp  �          @�
=���@:�H@#33A�{Cz����@�@Q�A��
C�                                    Bx\d~�  T          @�  ���H@HQ�?�(�A�33C�{���H@(�@1�AʸRCff                                    Bx\d�N  �          @�{���@<(�?��RA�Q�CG����@  @/\)A�=qCB�                                    Bx\d��  �          @�Q�����@E@�A�Q�C������@
=@:=qA�\)C                                      Bx\d��  �          @�z���
=@O\)@!�A��HC!H��
=@Q�@VffA�Q�Cp�                                    Bx\d�@  T          @�p�����@>�R@J�HA�p�Cff����?���@y��B{C�f                                    Bx\d��  �          @�������@4z�@P��A�G�C������?�G�@|(�B\)C                                     Bx\d֌  �          @�(���(�@(Q�@hQ�B�C����(�?�(�@��B�C#8R                                    Bx\d�2  �          @�(���G�@B�\@Y��A�33C����G�?�
=@�(�B�\C\                                    Bx\d��  �          @�=q��  @E�@S�
A��C\��  ?��R@��B  CL�                                    Bx\e~  �          @�Q�����@_\)@P��A�G�C������@��@�z�B33Cs3                                    Bx\e$  �          @Ϯ��
=@QG�@@  A�ffCG���
=@��@tz�Bp�Cn                                    Bx\e�  �          @�Q�����@]p�@-p�AĸRC33����@!G�@fffBG�CE                                    Bx\e.p  �          @�ff���\@Y��@?\)A�
=Cc����\@Q�@vffB�
C�                                    Bx\e=  �          @�(����@_\)@@��A�
=C�����@p�@y��B��C�
                                    Bx\eK�  �          @�=q�u@n{@6ffA�=qC��u@.{@s�
B  C��                                    Bx\eZb  �          @��H��@.{��z��`��C
��@AG������p�C�=                                    Bx\ei  �          @�{���\@   ?��RA��\C}q���\?���@
=qA�ffCp�                                    Bx\ew�  �          @����b�\@J�H?�(�A�  C
+��b�\@��@3�
B�RCW
                                    Bx\e�T  �          @��z�H@_\)?���AN=qC
O\�z�H@>�R@�A��HC�                                     Bx\e��  T          @�{�}p�@i��>���@^{C	c��}p�@X��?��Ao
=C}q                                    Bx\e��  �          @����\)@e�>�ff@��HC��\)@R�\?��RAx(�C(�                                    Bx\e�F  �          @������\@U�?�R@�Q�Cp����\@?\)?�=qA�  CW
                                    Bx\e��  �          @�Q��}p�@Z=q�W
=�CB��}p�@Tz�?O\)A�RC�                                    Bx\eϒ  �          @�  �a�@Y���k��*=qC\�a�@aG�>\)?ǮC�                                    Bx\e�8  �          @����XQ�@a녿�G��g
=C���XQ�@o\)�����C)                                    Bx\e��  �          @�
=�[�@XQ쿜(��c\)Ck��[�@e�����  C��                                    Bx\e��  �          @�\)�h��@U�������C	��h��@i�������  C��                                    Bx\f
*  T          @�z��G�@J=q�x���8�B���G�@��0������B�                                    Bx\f�  �          @�p���(�@6ff�����Pp�B����(�@�Q��N{���B�aH                                    Bx\f'v  �          @�  ��p�@/\)���
�c�RB�ff��p�@�Q��e��!��B�33                                    Bx\f6  �          @�����?��H��z��B˳3���@U��=q�I��B�                                      Bx\fD�  T          @���?�ff=��
������@4z�?�ff?˅��Q���B7{                                    Bx\fSh  �          @�=q?�\)>�ff����AxQ�?�\)?��������u��BIQ�                                    Bx\fb  �          @�=q?c�
?W
=��z��B-�?c�
@�\�����sz�B��{                                    Bx\fp�  �          @�z�?�\?����R��B�  ?�\@#33��(��g�B�aH                                    Bx\fZ  �          @�33���@z���33�B�\���@XQ��p  �?�\B���                                    Bx\f�   �          @���>��H?�\)��  33B�8R>��H@I����  �Z�HB�8R                                    Bx\f��  �          @�\)�\)@�R�����B�
=�\)@i����=q�@p�B�L�                                    Bx\f�L  �          @�z�J=q@ff�����z33B�#׿J=q@l���u��533B��                                    Bx\f��  T          @������@#33��  �Sz�B�uÿ��@i���AG��G�B�L�                                    Bx\fȘ  �          @����(�@��u�KQ�C\��(�@_\)�9����RB��H                                    Bx\f�>  �          @�
=�G�@*�H�C33�$�Cff�G�@^�R���ԣ�B�G�                                    Bx\f��  �          @�G���(�@��(��np�B�aH��(�@QG��Q��-�B��f                                    Bx\f�  �          @����H?������h�C^����H@H���W��-(�B�8R                                    Bx\g0  T          @���
=q?�p��y���Tp�C	� �
=q@E�E���B��f                                    Bx\g�  T          @��\�ff?��tz��Wz�C
�=�ff@;��C�
�!\)B�.                                    Bx\g |  �          @����O\)@G��$z���HC��O\)@=p��޸R����C	�f                                    Bx\g/"  �          @�
=�(Q�@G��H���(��C#��(Q�@H�������C�                                    Bx\g=�  �          @�p��QG�@�����C��QG�@=p���
=��\)C	�\                                    Bx\gLn  �          @�(��N�R@�����
=C��N�R@>{��  �~ffC	^�                                    Bx\g[  �          @��\�#33@p��>{�%�C
=�#33@A��Q���Q�C�                                    Bx\gi�  �          @���p  ?�\��=q�j{C� �p  @G�������HC��                                    Bx\gx`  �          @�����ff?
=��G���  C+�3��ff?��=��
?��C+�H                                    Bx\g�  �          @��
��p�?W
=?(�A=qC(�
��p�?(�?W
=A4��C+��                                    Bx\g��  �          @����z�?�Q�?��A�G�C ޸��z�?u?�
=A�=qC&�q                                    Bx\g�R  �          @���o\)@!G�>8Q�@33C��o\)@�?xQ�AEC\                                    Bx\g��  �          @��H�z�H?�{?���Ai�C�)�z�H?�z�?�z�A�  C @                                     Bx\g��  T          @�(��^�R@\)��������C@ �^�R@#�
�����p�C��                                    Bx\g�D  �          @�(��X��@�����Cu��X��@*=q�����ffC�)                                    Bx\g��  �          @�=q�Y��@
�H��p���G�Cc��Y��@!녿5��CO\                                    Bx\g�  A          @�z��`  @����\�YCJ=�`  @"�\�aG��=p�C\                                    Bx\g�6  
�          @�Q��p��@����C
=�p��@��>L��@'�Cc�                                    Bx\h
�  �          @����{�?��R�������CG��{�?�\)�fff�:=qC��                                    Bx\h�  �          @��R����?�
=��z�����CY�����@Q쿅��MG�C33                                    Bx\h((  �          @�G��hQ�=�\)�%�  C2���hQ�?fff���(�C&(�                                    Bx\h6�  T          @��\�n�R�B�\�%��
��C6ٚ�n�R?&ff� ����C*�                                    Bx\hEt  T          @����w����
�"�\���C50��w�?=p�����33C)33                                    Bx\hT  
�          @���P  ?�\)�k��c
=CxR�P  ?�=q��33���\C��                                    Bx\hb�  T          @��7
=@N�R�(����C���7
=@P  >�@���CQ�                                    Bx\hqf  �          @�����\@X�ÿ���n�HB�#���\@c�
=u?O\)B��\                                    Bx\h�  �          @�z��>{@mp��=p��ffC �H�>{@p��>��H@���C Q�                                    Bx\h��  T          @�33����@ �׿�\)��{C)����@ff�#�
��C�                                     Bx\h�X  �          @�33��  ?L�;���E�C*p���  ?W
=<#�
>#�
C)��                                    Bx\h��  �          @�=q����?�<#�
=uC-������?   >8Q�@C.�                                    Bx\h��  �          @����G�?@  >���@x��C+
=��G�?(�?\)@ҏ\C,Ǯ                                    Bx\h�J  "          @�(����?p��?�@ȣ�C(� ���?8Q�?O\)AG�C+aH                                    Bx\h��  
�          @�z���
=?���?z�@��HC&����
=?aG�?k�A-p�C)n                                    Bx\h�  �          @�(���(�?��?��AF�\C'J=��(�?&ff?�=qA}C,�                                    Bx\h�<  T          @�{��(�?�G�?B�\Ap�C!����(�?�
=?��HAc\)C%�                                    Bx\i�  "          @����\)?�Q�?��\A:�HC#!H��\)?��\?�Q�A�{C'��                                    Bx\i�  T          @�z����?�=q>��@�G�C����?���?��A<��C!�f                                    Bx\i!.  �          @�����?���>�33@�G�C� ���?�p�?xQ�A5�C�                                    Bx\i/�  
�          @�(�����?fff>�p�@�C)�\����?:�H?&ff@�G�C+��                                    Bx\i>z  �          @������?�33=�?���C@ ����?�G�?=p�A(�C�                                    Bx\iM   "          @��R��\)@
�H>�33@x��CaH��\)?�
=?�ffA9p�C�\                                    Bx\i[�  T          @������@�>��
@b�\Ck�����?�{?}p�A/33C��                                    Bx\ijl  T          @�Q�����@)���#�
��G�C������@#33?8Q�@��C��                                    Bx\iy  �          @�����p�@����Ϳ��C����p�@z�?5@�G�C��                                    Bx\i��  �          @�  ����@������{Cn����@?(��@���CJ=                                    Bx\i�^  �          @��R����@
==���?�=qC{����@(�?aG�A�RC��                                    Bx\i�  "          @�
=����@�
>�
=@�=qC������?�ff?���A>ffCQ�                                    Bx\i��  �          @������@ff?�@�G�C8R����?�?���AS\)CB�                                    Bx\i�P  "          @�
=���@Q�?\)@�z�C�����?�=q?��RA[33C�f                                    Bx\i��  T          @�=q��\)@�
?5@�C  ��\)?�Q�?�Q�A{\)C�R                                    Bx\iߜ  "          @�Q���(�@?Q�A�
C8R��(�?�
=?�ffA��CaH                                    Bx\i�B  
�          @������@��?���AR=qCE���?�\)?�A�{C ��                                    Bx\i��  
�          @�G���{@G�?�=qAiC����{?��H?�
=A�C"�3                                    Bx\j�  "          @�  ���
?��?�(�A��Cٚ���
?�{@{Aȏ\C&�=                                    Bx\j4  T          @����p�?�p�?�{A��\C"n��p�?E�@  A˅C*��                                    Bx\j(�  �          @��H��p�?˅@�A�33C!&f��p�?Q�@��Aڣ�C*\                                    Bx\j7�  �          @��H��  ?�ff@{A�\)C@ ��  ?p��@,��A�\)C(+�                                    Bx\jF&  �          @��
��p�?�z�@#�
A�p�Cc���p�?5@>{B(�C*�                                    Bx\jT�  
�          @�����ff?�(�@*�HA�
=C!�=��ff>�@@  BffC-�
                                    Bx\jcr  �          @�{����?���@1�A�ffC c�����?
=q@I��B��C-\                                    Bx\jr  �          @��R���H?�=q@A�B�RC#����H>aG�@S33B��C1
=                                    Bx\j��  �          @�\)��\)?�@3�
A�\)C"s3��\)>\@HQ�BG�C/!H                                    Bx\j�d  
�          @�Q���  ?�=q@8Q�A��
C#����  >�\)@J=qB�C0s3                                    Bx\j�
  �          @����z�?��@Mp�B\)C&���z�u@X��B�C4�                                    Bx\j��  �          @�����?.{@tz�B*(�C*������(��@tz�B*\)C=
                                    Bx\j�V  �          @��
���?�{@Z�HB�C"@ ���=�@j�HB#�C2p�                                    Bx\j��  �          @����{?��@p  B#�\C%u���{��  @y��B+ffC7u�                                    Bx\jآ  �          @��R��G�?\@s33B\)C }q��G�>\)@��HB.p�C20�                                    Bx\j�H  �          @���  >�Q�@q�B�C/^���  �s33@k�BC?�f                                    Bx\j��  T          @�
=��G�?p��@l(�B{C(Y���G��\@r�\B  C8Ǯ                                    Bx\k�  "          @�\)����?��\@>�RA�C%������>.{@O\)BG�C2
=                                    Bx\k:  �          @�\)��?�ff?�33A���C$���?O\)@z�A�33C+��                                    Bx\k!�  T          @����?�{@,��A�=qC!�3��?z�@EA�33C-@                                     Bx\k0�  
�          @�����?�G�@FffBG�C%����=�@VffB�HC2�\                                    Bx\k?,  6          @�(���Q�?�(�?�G�A~�\C!���Q�?���@ ��A�
=C'��                                    Bx\kM�  @          @�{����?��H?O\)A�
C.����?Ǯ?�Q�AyC"L�                                    Bx\k\x  	�          @�
=���
?�@A�(�C ����
?Y��@!�A���C)��                                    Bx\kk  T          @�����z�?���    �#�
C(33��z�?��>Ǯ@���C(�f                                    Bx\ky�  �          @������?޸R�.{���HC!W
����?�{����Q�C 
                                    Bx\k�j  "          @�p����\@G���\)�k33CG����\@��
=�ȣ�Cٚ                                    Bx\k�  T          @�  ���\@Q쿆ff�0��C���\@&ff�.{��(�C��                                    Bx\k��  
�          @�{��(�@(Q쿆ff�1�Ck���(�@5���\)�E�C�
                                    Bx\k�\  "          @�����G�@Dz�޸R��33CǮ��G�@^�R����=qCB�                                    Bx\k�  T          @����33@xQ�n{�=qC����33@}p�>��@��RC�                                    Bx\kѨ  "          @����p�@y�������-C����p�@�G�>���@Mp�C�f                                    Bx\k�N  "          @�z����@�Q쿞�R�B�RC�f���@�{>k�@G�Cc�                                    Bx\k��  "          @�p���Q�@XQ��
�H��=qC���Q�@{��\(��\)C	L�                                    Bx\k��  "          @�����\@E�����\z�C����\@W��L�Ϳ�(�C�                                    Bx\l@  �          @������\@L(��L�Ϳ   Cٚ���\@@��?��A#\)CW
                                    Bx\l�  �          @��H��(�@e�����RC���(�@]p�?�  A��C��                                    Bx\l)�  �          @Å����@XQ�>�{@L(�C@ ����@C33?�  Ac�C�                                    Bx\l82  
�          @��H��33@dz�?L��@�=qC����33@Dz�@   A��C�3                                    Bx\lF�  �          @�33��@QG�@��A�(�C
=��@z�@I��A�z�C�f                                    Bx\lU~  �          @�=q��{@A�@
=A�
=C#���{@ ��@QG�B�C�                                    Bx\ld$  T          @�����ff@3�
?��HA���Cc���ff?���@4z�A��C}q                                    Bx\lr�  
�          @�G�����@   @�A�G�C�����?���@:�HA��C"��                                    Bx\l�p  T          @\��Q�@{@A��HC!H��Q�?��@.�RA��C%��                                    Bx\l�  
�          @������@��?�p�A�ffC�����?Ǯ@,��A�=qC#��                                    Bx\l��  T          @�z���G�@��?��RA��C����G�?�ff@-p�AхC#�                                    Bx\l�b  �          @�p���33@�@   A�
=Cs3��33?�p�@,(�AθRC$��                                    Bx\l�  �          @�ff���H@z�@�A�Q�Cz����H?�
=@333A�\)C%
=                                    Bx\lʮ              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\l�T              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\l��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\l��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\mF              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\m�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\m"�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\m18              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\m?�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\mN�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\m]*              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\mk�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\mzv              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\m�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\m��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\m�h              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\m�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\mô              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\m�Z              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\m�               ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\m�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\m�L              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\n�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\n�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\n*>              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\n8�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\nG�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\nV0              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\nd�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\ns|              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\n�"              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\n��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\n�n              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\n�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\n��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\n�`              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\n�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\n�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\n�R              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\o�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\o�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\o#D              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\o1�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\o@�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\oO6              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\o]�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\ol�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\o{(              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\o��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\o�t              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\o�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\o��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\o�f              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\o�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\o�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\o�X              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\o��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\p�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\pJ              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\p*�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\p9�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\pH<              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\pV�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\pe�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\pt.              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\p��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\p�z              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\p�               ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\p��              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\p�l              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\p�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\pڸ              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\p�^              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\p�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\q�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\qP              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\q#�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\q2�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\qAB              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\qO�              ��O���O���O���O���O���O���O���O���O���O���O�                                  Bx\q^�            @Å��p�@G�?��Az{C:���p�?�G�@
=A�z�C$n                                   Bx\qm4  
Z          @����33@(�?��A���C��33?���@{A��\C%��                                    Bx\q{�  �          @��H��ff@{@\)A���C޸��ff?��R@9��A�C&��                                    Bx\q��  �          @����=q@��?��A�z�C}q��=q?�{@#�
A�C%��                                    Bx\q�&  �          @����\)@ ��?�=qAs
=C�H��\)?��
@p�A���C&��                                    Bx\q��  �          @�����Q�@��?�G�A@��C����Q�?\?��HA�\)C$�{                                    Bx\q�r  �          @������@�?��A.�\CT{���?��?�
=A��
C!��                                    Bx\q�  �          @����=q@�?��A�ffC�q��=q?�\)@{A�p�C%�)                                    Bx\qӾ  T          @�G���p�@	��@�RA�  Cff��p�?�
=@7�A�\)C'(�                                    Bx\q�d  
�          @�=q���H?���@�A���C����H?��
@,(�Aҏ\C)
=                                    Bx\q�
  "          @�������@Q�?�
=A���Cu�����?�\)@A�ffC%��                                    Bx\q��  �          @�=q���@z�?��HA^�HC�����?�\)@p�A�
=C#^�                                    Bx\rV  T          @Å���@G�?�33AT(�C� ���?���@��A�(�C#�                                    Bx\r�  �          @\���@
�H?�z�AV{Cp����?�  @
=A�ffC$��                                    Bx\r+�  
�          @��
���@�?�Q�AZ�RCz����?�33@
=A��C%ٚ                                    Bx\r:H  
�          @�33����@�?\Ag
=Cc�����?�\)@�A��C&\                                    Bx\rH�  �          @��H����@�\?���Ao33C������?���@p�A�ffC&�
                                    Bx\rW�  T          @��H���
?�ff?�  Ad  C":����
?�\)@�
A��C(��                                    Bx\rf:  �          @�G���33?�\)?�ffAn�\C#�{��33?p��@�\A�=qC*�                                     Bx\rt�  �          @�\)����?��R?�  A�{C$�)����?=p�@�A���C,T{                                    Bx\r��  �          @�ff���R?���?޸RA�z�C$  ���R?O\)@��A��C+��                                    Bx\r�,  �          @����Q�?�  @33A�(�C!�{��Q�?^�R@#33A�ffC*��                                    Bx\r��  "          @�z���33?�@{A�G�Ck���33?u@1G�A��C)Q�                                    Bx\r�x  �          @������@33@z�A���C������?�ff@:=qA�Q�C(33                                    Bx\r�  �          @�{����@�@
=A��C������?��@=p�A��C(\                                    Bx\r��  �          @����\)?���@�A�G�C �)��\)?Y��@2�\A�=qC*�{                                    Bx\r�j  �          @�\)���?��@��A�=qC$����?#�
@#�
A���C-8R                                    Bx\r�  �          @�\)��ff?�(�@��A�{C'Y���ff>�33@(�A���C0^�                                    Bx\r��  �          @�������?�{?���A�G�C&.����?
=q@�
A���C.^�                                    Bx\s\  �          @�Q����H?���?���A��C'�H���H>�(�@
=qA���C/��                                    Bx\s  �          @�����ff?�G�?ٙ�A��RC)����ff>��R?���A���C0��                                    Bx\s$�  �          @������?aG�?�33AV�RC+n���>���?У�Ay�C1�                                    Bx\s3N  �          @�����?��?�
=A��C&�f���?!G�@�
A��C-��                                    Bx\sA�  �          @�(���  ?���?���Ar�\C&u���  ?333@   A�ffC-
                                    Bx\sP�  �          @�33��  ?��?��
Ai�C'(���  ?&ff?�z�A�  C-�                                    Bx\s_@  �          @ə���Q�?ٙ�?�z�A���C#�\��Q�?aG�@��A���C+W
                                    Bx\sm�  �          @�33��  ?�@   A���C"���  ?s33@!G�A�p�C*��                                    Bx\s|�  �          @��
����?�ff?��RA��
C"������?p��@ ��A�  C*��                                    Bx\s�2  �          @�(���{?�=q@�RA�p�C"(���{?aG�@0  A��C+0�                                    Bx\s��  
�          @�z����@ff@�A�CT{���?�=q@>{Aݙ�C)�                                    Bx\s�~  �          @�33����@33@ffA��C������?��@<(�A���C)L�                                    Bx\s�$  �          @ə���
=?�\)@
=A�z�C$5���
=?:�H@#�
A��C,�                                    Bx\s��  T          @ə����?˅@�
A���C$�����?5@   A�33C,�f                                    Bx\s�p  �          @�ff��z�?�
=@�A��C%����z�?��@   A�(�C.ff                                    Bx\s�  �          @��
���?�Q�@ffA���C%xR���?\)@\)A��
C.33                                    Bx\s�  �          @�����Q�?�Q�?���AUG�C �H��Q�?��@   A��C&�\                                    Bx\t b  �          @�
=��\)?У�?��HA�G�C#k���\)?c�
@(�A�ffC*�\                                    Bx\t  �          @�Q�����?�(�@�
A�
=C������?���@(��A�G�C(��                                    Bx\t�  �          @�����?ٙ�@�
A��HC"k����?Q�@"�\A���C+L�                                    Bx\t,T            @�G�����?��?��AmG�C!.����?�
=@Q�A�G�C'�                                    Bx\t:�  T          @�����
=?�\?��A�(�C"���
=?}p�@z�A��HC)�q                                    Bx\tI�  �          @������H?���?�  A��C&{���H?#�
@��A���C-p�                                    Bx\tXF  �          @������?�  ?�
=A�  C'Q�����>�ff@  A�ffC/Y�                                    Bx\tf�  "          @����33?��
?�
=A�Q�C'!H��33>�@��A�\)C/
                                    Bx\tu�  T          @Å��?���?�G�A��
C&O\��?#�
@��A�ffC-�\                                    Bx\t�8  T          @�����?�(�?�33Aw\)C%������?B�\@z�A��RC,�                                     Bx\t��  �          @��
��?��R?ٙ�A�p�C%Y���?B�\@Q�A��C,n                                    Bx\t��  �          @�=q���
?�G�?��HA���C$�R���
?G�@��A�\)C,!H                                    Bx\t�*  �          @�=q����?˅?��
Ai��C$J=����?k�@ ��A�z�C*�=                                    Bx\t��  �          @\��
=?��?��
Ah��C&T{��
=?:�H?�
=A�{C,�3                                    Bx\t�v  T          @������?У�?�ffAp��C#������?s33@�\A��C*L�                                    Bx\t�  �          @�����H?���?���Aw�
C&{���H?333?��RA��
C,�
                                    Bx\t��  �          @�  ���?�?޸RA���C%�����?.{@��A��C,�q                                    Bx\t�h  T          @�����?��H?���A{
=C%&f����?E�@�A�  C,�                                    Bx\u  
�          @�p����?�G�?���A{�C$�����?Q�@33A�C+��                                    Bx\u�  S          @�{��=q?�
=?\Alz�C%�=��=q?G�?�Q�A�(�C,�                                    Bx\u%Z  �          @�  ��G�?���?�z�A�ffC#���G�?aG�@Q�A�(�C+                                      